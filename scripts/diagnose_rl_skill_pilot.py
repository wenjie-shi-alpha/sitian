"""Read-only diagnosis of the frozen pilot; never rewrites its endpoint metrics."""
import json,re,sys,statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sitian.case import CaseBundle
from sitian.env import ForecastEnv
P=ROOT/'data/experiments/rl_skill_pilot_20260909'
protocol=json.loads((P/'protocol.json').read_text());paths={c['case_id']:Path(c['case_dir']) for c in protocol['cases']}
report={'arms':{},'limitations':['Decoded-text tool-call extraction is diagnostic; malformed JSON is not repaired.','No training trajectories were saved: group-wise reward ranking and retained unique case coverage cannot be reconstructed from aggregate logs.','Conditional common-success comparisons do not identify a causal effect.']}
arms={}
for step in [0,25,50]:
 rows=[json.loads(l) for l in (P/f'validation/{step}.jsonl').read_text().splitlines()];out=[];tool_episodes=Counter()
 for row in rows:
  rec=json.loads(row['sitian_record']);blocks=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',row['output'],re.S)
  actions=[];errors=[]
  for index,block in enumerate(blocks):
   try:
    a=json.loads(block);args=a.get('arguments',{})
    if isinstance(args,str):args=json.loads(args)
    actions.append({'name':a['name'],'args':args})
   except (ValueError,TypeError,KeyError) as exc:errors.append({'index':index,'error':str(exc)[:140]})
  tool_episodes.update(set(a['name'] for a in actions))
  env=ForecastEnv(CaseBundle.load(paths[rec['case_id']]));env.reset();replayed=None;tool_errors=[]
  for a in actions:
   if env.done:break
   obs,reward,done,info=env.step(a)
   if not obs['ok'] or obs.get('content',{}).get('accepted') is False:tool_errors.append({'tool':a['name'],'error':str(obs['content'])[:220]})
   if info.get('reason')=='submitted':replayed=info['score']
  matched= bool(replayed and abs(replayed['composite']-row['score'])<.00011 and abs(replayed['outcome_composite']-row['outcome_composite'])<.00011) if row['sitian_submitted'] else None
  out.append({'case_id':rec['case_id'],'submitted':row['sitian_submitted'],'outcome':row['outcome_composite'],'reward':row['score'],'components':rec['components'],'response_tokens':rec['response_tokens'],'assistant_tokens':rec['assistant_tokens'],'tool_calls':len(blocks),'tools':dict(Counter(a['name'] for a in actions)),'json_errors':errors,'tool_errors':tool_errors,'accepted_replay_matches_saved_score':matched,'offline_replay_outcome':replayed['outcome_composite'] if replayed else None})
 arms[step]={x['case_id']:x for x in out}
 report['arms'][str(step)]={'tool_episode_counts':dict(tool_episodes),'mean_assistant_tokens':statistics.mean(x['assistant_tokens'] for x in out),'submitted_replay_matched':sum(x['accepted_replay_matches_saved_score'] is True for x in out),'submitted_replay_mismatched':sum(x['accepted_replay_matches_saved_score'] is False for x in out),'failed_cases':[x for x in out if not x['submitted']],'cases':out}
common=[cid for cid,x in arms[0].items() if x['submitted'] and arms[50][cid]['submitted']]
comp={}
for key in arms[0][common[0]]['components']:
 ids=[cid for cid in common if arms[0][cid]['components'].get(key) is not None and arms[50][cid]['components'].get(key) is not None]
 if ids:comp[key]={'same_active_cases':len(ids),'before':statistics.mean(arms[0][cid]['components'][key] for cid in ids),'after':statistics.mean(arms[50][cid]['components'][key] for cid in ids)}
report['common_component_means']=comp
report['worst_common_cases']=sorted([{'case_id':cid,'before':arms[0][cid]['outcome'],'after':arms[50][cid]['outcome'],'delta':arms[50][cid]['outcome']-arms[0][cid]['outcome'],'tools_before':arms[0][cid]['tools'],'tools_after':arms[50][cid]['tools']} for cid in common],key=lambda x:x['delta'])[:8]
metrics={}
log=(P/'train.log').read_text()
for line in log.splitlines():
 m=re.search(r'training/global_step:(\d+)',line)
 if not m:continue
 step=int(m[1]);d={}
 for key in ['actor/grad_norm','actor/kl_loss','actor/pg_clipfrac','actor/lr','timing_s/update_weights','response_length/clip_ratio','training/off_policy/trajectory_staleness/mean']:
  v=re.search(re.escape(key)+r':(?:np.float64\()?([-+0-9.eE]+)',line)
  if v:d[key]=float(v[1])
 metrics[step]=d
report['training']={'steps_found':len(metrics),'metrics':metrics,'ranges':{key:{'min':min(v[key] for v in metrics.values() if key in v),'max':max(v[key] for v in metrics.values() if key in v)} for key in next(iter(metrics.values()))}}
(P/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['arms','training']},ensure_ascii=False,indent=2))
for step,r in report['arms'].items():
 print(step,r['tool_episode_counts'],'replay matches',r['submitted_replay_matched'],'mismatches',r['submitted_replay_mismatched'])
 for x in r['failed_cases']:print(x['case_id'],'calls',x['tool_calls'],'json_errors',len(x['json_errors']),'offline score',x['offline_replay_outcome'])
print('TRAIN',report['training']['ranges'])
