#!/usr/bin/env python3
"""Produce independent exclusion/candidate manifests; preserve original split membership."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,sys
from datetime import date,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sitian.admission import exclusion_reasons,publication_assessment
from sitian.case import CaseBundle
from sitian.data_contract import issue_time


def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=1)+'\n')

def run(handoff,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    delivery=read(handoff/'DELIVERY_MANIFEST.json')
    checked=[]
    for rel in ('quality/all_case_quality.jsonl.gz','evidence/manifest.json','regional/manifest.json'):
        record=next(x for x in delivery['files'] if x['path']==rel)
        assert sha(handoff/rel)==record['sha256'],rel
        checked.append(record)
    with gzip.open(handoff/'quality/all_case_quality.jsonl.gz','rt') as f:quality=[json.loads(x) for x in f]
    qmap={(r['city'],r['issue_date']):r for r in quality};assert len(qmap)==len(quality)
    gaps=collections.defaultdict(set)
    for r in read(handoff/'evidence/manifest.json')['gaps']:gaps[r['issue_date']].add(r['kind'])
    episodes={}
    for split in ('train','val','test'):
        for ep in read(ROOT/f'data/interim/episodes_{split}.json'):
            k=ep['city'],ep['issue_date'];assert k not in episodes
            episodes[k]={**ep,'split':split,'path':f'cases/national/{split}/{ep["city"]}_{ep["issue_date"]}'}
    assert set(episodes)==set(qmap)
    included=[];excluded=[];publication=[];time_failures=[];pub_counts=collections.Counter();optional=collections.Counter()
    for k,q in sorted(qmap.items()):
        ep=episodes[k];reasons=exclusion_reasons(q,gaps[q['issue_date']])
        row={'city':k[0],'issue_date':k[1],'split':ep['split'],'reference_case':ep['path'],
             'reasons':reasons,'raw_truth_valid':not q['truth_failures'],
             'missing_required_evidence':sorted(gaps[q['issue_date']]),'training_ready':False}
        if reasons:excluded.append(row);continue
        b=CaseBundle.load(ROOT/ep['path']);violations=b.audit_time_gate()
        if violations:
            time_failures.append({'case':ep['path'],'violations':violations})
            excluded.append({**row,'reasons':['recorded_time_gate_failure']});continue
        cutoff=issue_time(b.issue_date);status=collections.Counter()
        for block in b.observations.values():status[publication_assessment(block,cutoff)['status']]+=1
        for block in b.guidance.get('sources',{}).values():status[publication_assessment(block,cutoff)['status']]+=1
        def visit(x):
            if isinstance(x,dict):
                if 'available_at' in x:status[publication_assessment(x,cutoff,basis='legacy_fixed_latency')['status']]+=1
                for v in x.values():visit(v)
            elif isinstance(x,list):
                for v in x:visit(v)
        visit(b.evidence)
        pub_counts.update(status)
        regional=sorted(set(b.guidance.get('sources',{}))-{'cams'})
        optional.update(regional)
        included.append({**row,'reference_time_gate_passed_under_recorded_times':True,
                         'regional_sources_in_reference':regional,
                         'publication_status_counts':dict(status),
                         'actual_publication_verified':False,
                         'candidate_kind':'passes_existing_base_gates_for_rebuild'})
        publication.append({'case':ep['path'],'counts':dict(status),'strict_actual_publication_passed':False})
    keep={r['reference_case'] for r in included}
    valid_union=set()
    for split in ('train','val','test'):valid_union.update(read(ROOT/f'data/interim/valid_cases_{split}.json'))
    assert keep==valid_union,'Exclusion result differs from prior independently validated population'
    names={'train':'train_without_winter_or_spatial_holdout','val':'valid_cases_val','winter_selection':'challenge_winter','test':'valid_cases_test','spatial_ood_val':'spatial_ood_val','spatial_ood_test':'spatial_ood_test'}
    frozen={};frozen_receipts=[]
    for label,name in names.items():
        path=ROOT/f'data/interim/{name}.json';rows=read(path);filtered=[r for r in rows if r in keep]
        assert filtered==rows,(label,'unexpected split member removed')
        write(out/f'candidate_splits/{label}.json',filtered)
        frozen[label]=filtered;frozen_receipts.append({'name':label,'source':str(path.relative_to(ROOT)),'sha256':sha(path),'cases':len(rows),'membership_unchanged':True})
    # Use only dates and city identities; no model predictions or outcomes guide selection.
    case_metadata={p:read(ROOT/p/'case.json') for p in keep}
    def targets(paths):return {(case_metadata[p]['region'],str(date.fromisoformat(case_metadata[p]['issue_date'])+timedelta(days=i))) for p in paths for i in range(1,int(case_metadata[p]['horizon'])+1)}
    train=targets(frozen['train']);overlaps={name:len(train&targets(frozen[name])) for name in ('val','winter_selection','test')}
    holdout=read(ROOT/'data/interim/spatial_ood_audit.json')['heldout_cities'];cities={case_metadata[p]['region'] for p in frozen['train']}
    assert not any(overlaps.values());assert not (cities&set(holdout))
    write(out/'candidate_pool.json',[r['reference_case'] for r in included]);write(out/'excluded_cases.json',excluded)
    with gzip.open(out/'candidate_audit.jsonl.gz','wt',encoding='utf-8') as f:
        for r in included:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    write(out/'publication_audit.json',publication);write(out/'strict_publication_certified_cases.json',[])
    policy={
      'version':'candidate-admission-v1','intended_use':'candidate population for receiver-side rebuild, not training authorization',
      'required':['audited six-pollutant truth','existing input-count gate','CAMS native series','four open-evidence channels'],
      'regional_modes':{'required':False,'on_missing':'explicit absence; no zero filling','comparisons':'reuse identical per-case source set and missingness masks for frozen, RL and baseline arms'},
      'publication':{'legacy_latency_hours':{'cams':10,'gfs':5,'ifs':8},'legacy_observation_input_buffer_hours':1,'actual_publication_verified':False,'timestamps_must_not_be_backfilled_as_observations':True,'historical_schedule_applicability':'must be documented separately; current documentation is not a per-run historical receipt'},
      'guidance':{'D5':'keep absent and visible as missing','CAMS_O3':'3-hourly instantaneous proxy; never relabel as 8-hour target'},
      'training_ready':False,'remaining_gates':['receiver rebuild from native values and repaired code','publication assumption/evidence protocol','bias/history index and Parquet identities','runtime acceptance','fresh training under hold policy']}
    write(out/'policy.json',policy)
    summary={'requested':len(quality),'excluded_unique':len(excluded),'candidate_pool':len(included),'exclusion_reason_counts_nonexclusive':dict(collections.Counter(r for x in excluded for r in x['reasons'])),'candidate_counts_by_original_split':dict(collections.Counter(r['split'] for r in included)),'frozen_splits':frozen_receipts,'target_city_day_overlap':overlaps,'heldout_cities_in_train':sorted(cities&set(holdout)),'recorded_time_gate_failures':time_failures,'publication_record_statuses':dict(pub_counts),'actual_publication_certified_cases':0,'optional_regional_case_counts':dict(optional),'training_ready':False,'sources':checked}
    write(out/'summary.json',summary)
    for r in frozen_receipts:assert sha(ROOT/r['source'])==r['sha256']
    print(json.dumps({k:v for k,v in summary.items() if k not in ('frozen_splits','sources')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.handoff.resolve(),a.out.resolve())
