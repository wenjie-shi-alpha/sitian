#!/usr/bin/env python3
"""Rebuild only the audited missing cases into a new isolated workspace.

Never repoints training or overwrites original raw data, cases, DB or splits.
Full evidence is required before any recovered case is marked training-ready.
"""
from __future__ import annotations
import argparse,collections,importlib.util,json,shutil,sqlite3,subprocess,sys
from datetime import date
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_unbuilt_local import cache_store,dump,identity,load_module,sha

def run(root,audit,reference,out):
    out.mkdir(parents=True,exist_ok=False)
    workspace=out/'workspace';workspace.mkdir()
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    for d in ('scripts','src','tests','configs','references'):
        shutil.copytree(root/d,workspace/d,ignore=ignore)
    for f in ('pyproject.toml','conftest.py','README.md'):shutil.copy2(root/f,workspace/f)
    interim=workspace/'data/interim';interim.mkdir(parents=True)
    for name in ('city_coords.json','climatology.json'):
        shutil.copy2(root/'data/interim'/name,interim/name)
    corrected=[]
    for group in ('scripts','src','tests'):
        for p in (reference/group).rglob('*.py'):
            dest=workspace/p.relative_to(reference);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            corrected.append(identity(dest))
    dump(out/'corrected_code_identity.json',corrected)
    # Validate the supplied bug fixes before using the copy.
    tests=['tests/test_meteorology.py','tests/test_data_snapshot_repair.py','tests/test_spatial_ood.py']
    with (out/'corrected_code_tests.log').open('w') as f:
        subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=workspace,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Corrected-code regression tests passed.',flush=True)
    sys.path.insert(0,str(workspace/'src'))
    b=load_module(workspace/'scripts/build_national_case.py','corrected_builder')
    daily=load_module(workspace/'scripts/build_daily_all.py','strict_daily')
    b.RAW_OBS=root/'data/raw/aq_obs/cities';b.CAMS_DIR=root/'data/raw/cams'
    dbpath=workspace/'data/aq_daily.sqlite'
    db=sqlite3.connect(dbpath)
    db.execute('CREATE TABLE daily (city TEXT,date TEXT,pm25 REAL,pm10 REAL,o3_8h REAL,so2 REAL,no2 REAL,co REAL,n_hours INTEGER,PRIMARY KEY(city,date))')
    obs_files=json.loads((audit/'observation_file_inventory.json').read_text())
    for p in obs_files:db.executemany('INSERT OR REPLACE INTO daily VALUES (?,?,?,?,?,?,?,?,?)',daily.process_file(Path(p['path'])))
    db.commit();db.close();db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
    b.DB=dbpath
    findings=json.loads((audit/'case_findings.json').read_text())
    keys={r['case_id']:r for r in findings}
    # Representatives include each missing month and each present-day outcome.
    representatives={}
    for r in findings:
        k=(r['issue_date'][:7],tuple(r['current_blockers']))
        representatives.setdefault(k,r)
    store=cache_store(b.CamsStore());comparison=b.CamsStore()
    samples=[]
    for r in representatives.values():
        ep={k:r[k] for k in ('city','issue_date','stratum','cluster','split')}
        issue=date.fromisoformat(r['issue_date'])
        expected=comparison.extract(r['city'],issue);actual=store.extract(r['city'],issue)
        assert expected==actual, 'Cached and original extraction differ'
        b.REPO_ROOT=out/'samples'
        status=b.build_one(store,db,ep,r['split'])
        samples.append({'case_id':r['case_id'],'status':status,'cached_extraction_matches_original':True,'expected_blockers':r['current_blockers']})
    for ds in comparison._cache.values():ds.close()
    dump(out/'sample_validation.json',samples)
    print('Representative sample validation:',samples,flush=True)
    from sitian.case import CaseBundle
    from sitian.meteorology import wind_direction_label
    b.REPO_ROOT=workspace
    results=[];casepaths=[]
    for n,r in enumerate(findings):
        ep={k:r[k] for k in ('city','issue_date','stratum','cluster','split')}
        try:status=b.build_one(store,db,ep,r['split'])
        except Exception as ex:status='error';error=repr(ex)
        row={'case_id':r['case_id'],'split':r['split'],'status':status,'prior_blockers':r['current_blockers']}
        if status=='error':row['error']=error
        if status=='ok':
            assert r['can_rebuild_with_raw_valid_truth'], 'Unapproved case accepted'
            p=workspace/f'cases/national/{r["split"]}/{r["case_id"]}'
            bundle=CaseBundle.load(p)
            violations=bundle.audit_time_gate()
            assert not violations,(r['case_id'],violations)
            assert all(d['wind_dir']==wind_direction_label(d['wind_dir_deg']) for d in bundle.diagnostics['daily'].values())
            for f in p.glob('*.json'):json.dumps(json.loads(f.read_text()),allow_nan=False)
            assert b.read_truth(db,r['city'],date.fromisoformat(r['issue_date']))==bundle.truth
            row.update(time_gate_passed=True,wind_labels_passed=True,truth_matches_new_strict_db=True)
            row['full_open_evidence_present']=False
            row['training_ready']=False
            casepaths.append(str(p))
        else:assert not r['can_rebuild_with_raw_valid_truth'],(r['case_id'],status)
        results.append(row)
        if (n+1)%100==0:print('Rebuild progress',n+1,dict(collections.Counter(x['status'] for x in results)),flush=True)
    for ds in store._cache.values():ds.close()
    db.close()
    dump(out/'rebuild_results.json',results)
    dump(out/'recovered_base_cases.json',casepaths)
    dump(out/'rebuilt_file_identities.json',[identity(p) for d in casepaths for p in sorted(Path(d).glob('*.json'))])
    all_issues=sorted({r['issue_date'] for r in findings});evidence=[]
    for eroot in (Path('/root/sitian_open_evidence'),Path('/mnt/wsl/EAGET/sitian_open_evidence')):
        manifest=eroot/'manifests/open_evidence_v1.json';m=json.loads(manifest.read_text())
        evidence.append({'root':str(eroot),'manifest_identity':identity(manifest),'missing_issue_dates_from_manifest':sorted(set(all_issues)-set(m['issue_dates'])),'missing_derived_files':[f'{d}.{kind}.json' for d in all_issues for kind in ('synoptic','composition','fires','spatial_obs') if not (eroot/'derived/by_issue'/f'{d}.{kind}.json').is_file()]})
    dump(out/'evidence_gap_inventory.json',evidence)
    summary={'counts':dict(collections.Counter(x['status'] for x in results)),'samples':len(samples),'corrected_code_tests_passed':True,'successful_case_time_gate_failures':sum(not x.get('time_gate_passed') for x in results if x['status']=='ok'),'all_generated_files_hashed':True,'new_daily_db':identity(dbpath),'training_ready':False,'training_ready_reason':'None of the 33 issue dates was included in either existing open-evidence manifest; 132 expected derived files absent in each evidence root. Base cases must not be used as full-contract RL data.','original_dataset_changed':False,'active_training_changed':False}
    dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.root.resolve(),a.audit.resolve(),a.reference.resolve(),a.out.resolve())
