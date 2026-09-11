#!/usr/bin/env python3
"""Record local evidence gaps and verify frozen/source identities after rebuilding."""
from __future__ import annotations
import argparse,collections,json,sys
from datetime import datetime,timezone
from pathlib import Path
from audit_unbuilt_local import dump,identity,load_module,sha

def run(root,auditroot):
    result=auditroot/'results'; rebuild=auditroot/'rebuild';sys.path.insert(0,str(root/'src'))
    findings=json.loads((result/'case_findings.json').read_text());issues=sorted({r['issue_date'] for r in findings})
    from sitian.open_evidence import build_download_manifest,parse_utc
    manifest=build_download_manifest(issues)
    dump(result/'missing_issues_evidence_plan.json',manifest)
    fetch=load_module(root/'scripts/fetch_open_nwp.py','nwp_paths_only')
    roots=[Path('/root/sitian_open_evidence'),Path('/mnt/wsl/EAGET/sitian_open_evidence'),Path('/root/sitian_cams_recovery'),Path('/root/sitian_evidence_workspace')]
    summaries=[];availability=[]
    for eroot in roots:
        missing=[f'{d}.{k}.json' for d in issues for k in ('synoptic','composition','fires','spatial_obs') if not (eroot/'derived/by_issue'/f'{d}.{k}.json').is_file()]
        record={'root':str(eroot),'derived_present':132-len(missing),'derived_missing':len(missing)}
        mp=eroot/'manifests/open_evidence_v1.json'
        if mp.is_file():
            old=json.loads(mp.read_text());record.update(manifest=identity(mp),missing_issue_dates=sorted(set(issues)-set(old['issue_dates'])))
        summaries.append(record)
    # Use precisely the existing downloader's path mapping, without any downloads.
    for product,group in [('synoptic',manifest['nwp']),('radiation',manifest['nwp_radiation']['records'])]:
        for source,records in group.items():
            requested={}
            for r in records:
                _,_,filename=fetch._urls(source,r['cycle'],int(r['step_hour']))
                if product=='radiation':filename=filename.replace('.grib2','.radiation.grib2')
                base='nwp' if product=='synoptic' else 'nwp_radiation'
                rel=Path('raw')/base/source/parse_utc(r['cycle']).strftime('%Y%m%d%H')/filename
                requested.setdefault(str(rel),[]).append(r['issue_date'])
            for rel,ds in requested.items():
                matches=[]
                for eroot in roots:
                    p=eroot/rel; side=p.with_suffix(p.suffix+'.json')
                    if p.is_file():
                        m=json.loads(side.read_text()) if side.is_file() else {}
                        matches.append({'path':str(p),'bytes':p.stat().st_size,'sidecar_present':side.is_file(),'sidecar_size_matches':p.stat().st_size==m.get('bytes'),'sha_not_rechecked':True})
                availability.append({'product':product,'source':source,'relative_path':rel,'issue_dates':sorted(set(ds)),'matches':matches})
    dump(result/'nwp_local_cache_inventory.json',availability)
    dump(result/'evidence_roots_inventory.json',summaries)
    gap_counts={}
    for product in ('synoptic','radiation'):
        for source in ('gfs','ifs'):
            records=[r for r in availability if r['product']==product and r['source']==source]
            gap_counts[product+'_'+source]={'unique_requested_files':len(records),'present_in_any_checked_root':sum(bool(r['matches']) for r in records),'missing_in_all_checked_roots':sum(not r['matches'] for r in records)}
    dump(result/'nwp_local_cache_summary.json',gap_counts)
    # Positive control: the same path mapping must find a known built issue.
    baseline_root=roots[0]
    old_manifest=json.loads((baseline_root/'manifests/open_evidence_v1.json').read_text())
    control=[]
    for product,group in [('synoptic',old_manifest['nwp']),('radiation',old_manifest['nwp_radiation']['records'])]:
        for source,records in group.items():
            day=records[0]['issue_date'];records=[r for r in records if r['issue_date']==day];found=0
            for r in records:
                _,_,filename=fetch._urls(source,r['cycle'],int(r['step_hour']))
                if product=='radiation':filename=filename.replace('.grib2','.radiation.grib2')
                base='nwp' if product=='synoptic' else 'nwp_radiation'
                path=baseline_root/'raw'/base/source/parse_utc(r['cycle']).strftime('%Y%m%d%H')/filename
                found+=path.is_file()
            control.append({'product':product,'source':source,'issue_date':day,'requested':len(records),'found':found})
            assert found==len(records), 'Path mapping positive control failed'
    dump(result/'nwp_path_positive_control.json',control)
    print('Evidence caches:',summaries,flush=True);print('NWP coverage:',gap_counts,flush=True)
    # No missing data is restored based only on a filename or an old DB value.
    sources=json.loads((result/'source_identities.json').read_text());guard=json.loads((result/'frozen_original_guard.json').read_text())
    changed=[]
    for r in {x['path']:x for x in sources+guard}.values():
        p=Path(r['path'])
        if not p.is_file() or sha(p)!=r['sha256']:changed.append(str(p))
    receipt={'checked_source_identities':len(sources),'checked_frozen_files':len(guard),'changed_files':changed,'passed':not changed,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    dump(result/'preservation_verification.json',receipt)
    assert not changed,changed
    rebuilt=json.loads((rebuild/'rebuilt_file_identities.json').read_text())
    mismatches=[x['path'] for x in rebuilt if sha(Path(x['path']))!=x['sha256']]
    assert not mismatches,mismatches
    dump(rebuild/'output_hash_verification.json',{'files_checked':len(rebuilt),'changed_files':mismatches,'passed':True})
    print('Preservation and outputs:',receipt,'rebuilt files',len(rebuilt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audit-root',type=Path,required=True);a=ap.parse_args();run(a.root.resolve(),a.audit_root.resolve())
