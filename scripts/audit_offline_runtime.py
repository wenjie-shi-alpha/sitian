#!/usr/bin/env python3
"""Real veRL CPU lifecycle, dataset/token contracts and fresh-run configuration."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.tools.tool_registry import initialize_tools_from_config

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sitian.case import CaseBundle
from sitian.data_contract import issue_time, timestamp
from sitian.env import ForecastEnv
from sitian.provenance import file_identity
from audit_verl_runtime_contract import _assistant, _lifecycle, _config_checks, PINNED_VERL_COMMIT
from audit_trajectory_mask import _direct_verl_response_mask
from rebuild_native_snapshot import write


def make_config(out, environment):
    assets=out/"assets"
    environ={**os.environ, **environment, "SITIAN_CONFIG_ONLY":"1", "SITIAN_N_GPUS":"4",
        "FH_TRAINABLE_MODEL":str(ROOT/"models/Qwen3-8B"),
        "SITIAN_TRAIN_BATCH_SIZE":"8", "SITIAN_AGENT_WORKERS":"4", "SITIAN_SMOKE_STEPS":"50",
        "SITIAN_TOOL_CONFIG":str(assets/"tools.yaml"), "SITIAN_TRAIN_FILE":str(assets/"train.parquet"),
        "SITIAN_VAL_FILE":str(assets/"val.parquet"),
        "SITIAN_SMOKE_CKPT_DIR":str(ROOT/"data/checkpoints/offline_native_v3_20260912_fresh"),
        "SITIAN_EXPERIMENT_NAME":"offline_native_v3_fresh_pilot", "SITIAN_ROLLOUT_GPU_UTIL":"0.45",
        "SITIAN_ROLLOUT_MAX_NUM_SEQS":"8"}
    result=subprocess.run(["bash",str(ROOT/"scripts/run_verl_smoke.sh")],env=environ,cwd=ROOT,text=True,capture_output=True,check=True)
    (out/"audits/config_resolve.stderr.log").write_text(result.stderr)
    cfg=OmegaConf.create(yaml.safe_load(result.stdout))
    cfg.data.max_prompt_length=8192
    cfg.data.max_response_length=24576
    cfg.data.filter_overlong_prompts=False
    cfg.data.seed=20260912
    cfg.data.gen_batch_size=8
    cfg.data.apply_chat_template_kwargs={"enable_thinking":True}
    cfg.actor_rollout_ref.rollout.prompt_length=8192
    cfg.actor_rollout_ref.rollout.response_length=24576
    cfg.actor_rollout_ref.rollout.multi_turn.max_tool_response_length=26000
    cfg.trainer.resume_mode="disable"
    cfg.trainer.resume_from_path=None
    cfg.trainer.validation_data_dir=str(out/"future_training/validation")
    cfg.trainer.rollout_data_dir=str(out/"future_training/rollouts")
    cfg.trainer.test_freq=25
    cfg.trainer.val_before_train=True
    cfg.data.val_batch_size=8
    cfg.actor_rollout_ref.rollout.val_kwargs.n=1
    cfg.actor_rollout_ref.rollout.val_kwargs.do_sample=True
    cfg.actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    cfg.ray_kwargs.ray_init.runtime_env.env_vars.update(environment)
    OmegaConf.save(cfg,out/"train.yaml",resolve=True)
    # Validate the final configuration through the actual Hydra entrypoint.
    resolved=subprocess.run([sys.executable,"-m","verl.trainer.main_ppo","--config-path",str(out),
        "--config-name","train","--cfg","job","--resolve"],env={**os.environ,**environment},
        cwd=ROOT/".local/verl-upstream",text=True,capture_output=True,check=True)
    if OmegaConf.to_container(OmegaConf.create(yaml.safe_load(resolved.stdout)),resolve=True) != OmegaConf.to_container(cfg,resolve=True):
        raise ValueError("Hydra changed the final training configuration")
    return cfg


async def exercise_case(path, tools, tokenizer):
    bundle=CaseBundle.load(path)
    env=ForecastEnv(bundle);env.reset()
    allowed=set(env._tools)
    actions=[("list_data_assets",{}),("get_observations",{"last_hours":72,"stride":1}),
             ("get_process_evidence",{}),("get_native_meteorology",{}),
             ("compute_diffusion_conditions",{"weak_wind_ms":3.0}),
             ("get_synoptic_evidence",{"source":"gfs","valid_time":bundle.evidence["synoptic"]["sources"]["gfs"][3]["valid_time"],"detail":"full"}),
             ("get_pollution_evidence",{"kind":"composition","detail":"full","path":"/aerosol/records/0"}),
             ("retrieve_forecast_methods",{"query":"逆温 扩散 持续"}),
             ("get_guidance_bias",{"pollutant":"PM2.5"})]
    if "find_similar_cases" in allowed: actions.append(("find_similar_cases",{"top_k":2}))
    data=AgentData(messages=[],image_data=[],video_data=[],audio_data=None,mm_processor_kwargs=None,
                   metrics={},request_id=bundle.case_id,tools_kwargs={})
    rows=[]
    for name,args in actions:
        action={"name":name,"args":args}
        if name not in allowed: raise ValueError(f"required native tool unregistered: {name}")
        data.messages.append(_assistant(action,f"call-{len(rows)}"))
        start=time.monotonic()
        direct,reward,done,_=env.step(action)
        instance,_=await tools[name].create(create_kwargs={"case_dir":str(path),"harness_resources":env.resource_identity})
        try:
            response,actual_reward,metrics=await tools[name].execute(instance,args,agent_data=data)
        finally:
            await tools[name].release(instance)
        content=json.loads(response.text)
        if not direct["ok"] or content!=direct["content"] or reward!=actual_reward or done!=metrics["sitian_terminal"]:
            raise ValueError(f"native/direct parity mismatch {bundle.case_id}/{name}")
        if len(response.text)>24000: raise ValueError("tool response would be truncated")
        rows.append({"tool":name,"characters":len(response.text),"tokens":len(tokenizer.encode(response.text)),
                     "seconds_including_replay":round(time.monotonic()-start,3)})
        data.messages.append({"role":"tool","tool_call_id":f"call-{len(rows)-1}","content":response.text})
        if name=="find_similar_cases" and content.get("available"):
            for match in content["analogs"]:
                if timestamp(match["verification_available_at"]) >= issue_time(bundle.issue_date):
                    raise ValueError("future analog outcome")
            actions.append(("get_historical_case",{"case_id":content["analogs"][0]["case_id"],
                                                   "detail":"full","feature_prefix":"meteorology/"}))
    # Dataset/config mismatch must fail before a tool is created.
    try:
        instance,_=await tools["get_native_meteorology"].create(create_kwargs={"case_dir":str(path),"harness_resources":{}})
    except ValueError:
        rejected=True
    else:
        await tools["get_native_meteorology"].release(instance)
        rejected=False
    if not rejected: raise ValueError("resource mismatch accepted")
    return {"case_id":bundle.case_id,"calls":rows,"resource_mismatch_rejected":rejected}


def run(out):
    environment=json.loads((out/"runtime_environment.json").read_text());os.environ.update(environment)
    os.environ.update(SITIAN_TRAIN_BATCH_SIZE="8",SITIAN_SMOKE_STEPS="50")
    cfg=make_config(out,environment)
    bound_paths = set((ROOT/"src/sitian").rglob("*.py")) | set((out/"assets").glob("*"))
    bound_paths.update([out/"protocol.json", out/"runtime_environment.json", out/"train.yaml", Path(__file__),
                       ROOT/"scripts/audit_trajectory_mask.py", ROOT/"scripts/audit_verl_runtime_contract.py"])
    bound_files = [file_identity(path) for path in sorted(bound_paths) if path.is_file()]
    tokenizer=AutoTokenizer.from_pretrained(ROOT/"models/Qwen3-8B",local_files_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        configured=initialize_tools_from_config(str(out/"assets/tools.yaml"))
    tools={t.name:t for t in configured}
    schemas={name:t.tool_schema.model_dump(exclude_unset=True,exclude_none=True) for name,t in tools.items()}
    protocol=json.loads((out/"protocol.json").read_text())
    counts,max_prompt,representatives={},0,[]
    for dataset in sorted((out/"assets").glob("*.parquet")):
        table=pq.read_table(dataset).to_pylist()
        expected=json.loads((Path(protocol["snapshot"])/"manifests"/f"{dataset.stem}.json").read_text())
        if len(table)!=len(expected): raise ValueError("Parquet row loss")
        for i,(row,case_path) in enumerate(zip(table,expected)):
            info=row["extra_info"]
            if info["harness_resources"]!=protocol["resources"]: raise ValueError("dataset resources stale")
            for name in info["tool_selection"]:
                create=info["tools_kwargs"][name]["create_kwargs"]
                if (ROOT/create["case_dir"]).resolve()!=Path(case_path) or create["harness_resources"]!=protocol["resources"]:
                    raise ValueError("Parquet case/resource mismatch")
            prompt=tokenizer.apply_chat_template(row["prompt"],tools=[schemas[n] for n in info["tool_selection"]],
                tokenize=True,add_generation_prompt=True,enable_thinking=True)
            if isinstance(prompt, Mapping):
                prompt=prompt["input_ids"]
            if len(prompt) and isinstance(prompt[0], list):
                if len(prompt)!=1: raise ValueError("unexpected batched tokenizer response")
                prompt=prompt[0]
            if len(prompt)<100: raise ValueError("tokenizer did not return a token sequence")
            max_prompt=max(max_prompt,len(prompt))
            if len(prompt)>cfg.data.max_prompt_length: raise ValueError("prompt would be filtered/truncated")
            if i==0: representatives.append(Path(case_path))
        counts[dataset.stem]=len(table)
        print(f"Token/row check: {dataset.stem} {len(table)}, maximum prompt={max_prompt}",flush=True)
    # Include the earliest issue, where no verified history can exist.
    all_paths=json.loads((Path(protocol["snapshot"])/"manifests/all.json").read_text())
    earliest=min(all_paths,key=lambda p:Path(p).name.rsplit("_",1)[-1])
    representatives.append(Path(earliest))
    for city in ("拉萨", "北京", "广州"):
        matching=[p for p in all_paths if Path(p).name.startswith(city+"_")]
        if matching: representatives.append(Path(max(matching,key=lambda p:Path(p).name)))
    cases=[]
    for path in representatives:
        cases.append(asyncio.run(exercise_case(path,tools,tokenizer)))
        print(f"Native parity: {path.name}",flush=True)
    val=json.loads((Path(protocol["snapshot"])/"manifests/val.json").read_text())
    lifecycle=asyncio.run(_lifecycle(Path(val[0]),out/"assets/tools.yaml"))
    gas_lifecycle=asyncio.run(_lifecycle(Path(val[0]),out/"assets/tools.yaml",optional_gases=True))
    mask=_direct_verl_response_mask(ROOT/"models/Qwen3-8B")
    checks={**_config_checks(out/"train.yaml"),**lifecycle["checks"],
        **{"gas_"+key:value for key,value in gas_lifecycle["checks"].items()},"actual_verl_assistant_mask":mask["pass"],
        "pinned_verl":subprocess.check_output(["git","-C",str(ROOT/".local/verl-upstream"),"rev-parse","HEAD"],text=True).strip()==PINNED_VERL_COMMIT,
        "all_parquet_rows_preserved":counts==protocol["splits"],"fresh_run_only":cfg.trainer.resume_mode=="disable" and cfg.trainer.resume_from_path is None,
        "context_budget":cfg.data.max_prompt_length+cfg.data.max_response_length==cfg.actor_rollout_ref.rollout.max_model_len,
        "tool_character_limit":cfg.actor_rollout_ref.rollout.multi_turn.max_tool_response_length>protocol["resources"]["max_tool_response_chars"],
        "worker_resource_environment":all(cfg.ray_kwargs.ray_init.runtime_env.env_vars[k]==v for k,v in environment.items())}
    report={"passed":all(checks.values()),"checks":checks,"rows":counts,"maximum_prompt_tokens_with_tools":max_prompt,
        "bound_files": bound_files,
        "cases":cases,"lifecycle":lifecycle,"gas_lifecycle":gas_lifecycle,"mask":mask,"config":file_identity(out/"train.yaml"),
        "scope":"Real veRL BaseTool, Hydra and tokenizer/continuous-token mask; CPU only, no optimizer or model generation",
        "training_started":False}
    for expected in bound_files:
        if file_identity(expected["path"]) != expected:
            raise ValueError("runtime input or implementation changed during audit")
    write(out/"audits/runtime.json",report)
    print(json.dumps({"passed":report["passed"],"checks":checks,"maximum_prompt":max_prompt}),flush=True)
    if not report["passed"]: raise ValueError("runtime readiness checks failed")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True,type=Path);a=p.parse_args();run(a.out.resolve())
