#!/usr/bin/env python3
"""Export existing city evidence and traceable expert text without raw grids."""
from __future__ import annotations
import argparse,collections,csv,gzip,json,re,subprocess
from datetime import date,timedelta
from pathlib import Path
if __package__:
    from .export_native_city_data import identity
else:
    from export_native_city_data import identity


def write(stream,record):
    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


def city_subset(value,cities):
    if isinstance(value,list):return [city_subset(v,cities) for v in value]
    if isinstance(value,dict):
        return {k:({c:city_subset(x,cities) for c,x in v.items() if c in cities}
                   if k=='cities' and isinstance(v,dict) else city_subset(v,cities)) for k,v in value.items()}
    return value


def export_evidence(request,root,out):
    out.mkdir(parents=True,exist_ok=False);req=json.loads(request.read_text());byday=collections.defaultdict(set)
    for c in req['cases']:byday[c['issue_date']].add(c['city'])
    sources={};raw={};gaps=[];count=0;profile_count=0;levels=collections.defaultdict(set)
    with gzip.open(out/'existing_city_evidence.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as f:
        for day,cities in sorted(byday.items()):
            for kind in ('synoptic','composition','fires','spatial_obs'):
                p=root/'derived/by_issue'/f'{day}.{kind}.json'
                if not p.is_file():gaps.append({'issue_date':day,'kind':kind,'cities':sorted(cities),'reason':'derived_source_missing'});continue
                data=json.loads(p.read_text());ident=identity(p);sources[ident['sha256']]=ident
                write(f,{'issue_date':day,'requested_cities':sorted(cities),'source_sha256':ident['sha256'],'representation':'existing_derived_city_evidence','values_are_native':False,'availability_verified':False,'availability_note':'available_at inside legacy payload is fixed-latency policy metadata, not observed publication','data':city_subset(data,cities)})
                count+=1
                if kind=='synoptic':
                    for provider,rs in data['synoptic']['sources'].items():
                        for r in rs:
                            for city in cities:
                                values=r.get('cities',{}).get(city)
                                if values:
                                    profile_count+=1
                                    for field in ('temperature_c','relative_humidity_pct','wind','omega_pa_s'):levels[field].update(values.get(field,{}))
                            ref=r.get('raw') or {}
                            if ref.get('relative_path'):raw[(str(root),ref['relative_path'])]=ref
        static=root/'derived/static_context.json'
        if static.is_file():
            ident=identity(static);sources[ident['sha256']]=ident
            write(f,{'kind':'static_context','source_sha256':ident['sha256'],'representation':'existing_derived_city_evidence','data':city_subset(json.loads(static.read_text()),set(req['coordinates']))})
    with gzip.open(out/'raw_source_sidecars.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as f:
        for (r,rel),record in sorted(raw.items()):
            p=Path(r)/rel;side=p.with_suffix(p.suffix+'.json')
            row={'raw':record,'path':str(p),'exists':p.is_file(),'raw_sha256_rechecked':False,'actual_publication_verified':False}
            if p.is_file():row['actual_bytes']=p.stat().st_size;row['size_matches_record']=p.stat().st_size==record.get('bytes')
            if side.is_file():row.update(sidecar_identity=identity(side),sidecar=json.loads(side.read_text()))
            write(f,row)
    manifest={'request':identity(request),'extractor':identity(Path(__file__)),'evidence_root':str(root),'source_kind':'derived values retained exactly, not raw GRIB extraction','records':count,'city_profile_snapshots':profile_count,'pressure_levels_hpa':{k:sorted(v,key=float) for k,v in levels.items()},'inversion_limitation':'Only temperature at 925/850/700 hPa; cannot resolve a continuous or shallow surface inversion, and below-terrain levels need masking by receiver.','publication_limitation':'Legacy available_at values come from configured latency rules; actual publication not verified.','raw_grids_transferred':False,'sources':list(sources.values()),'raw_source_references':len(raw),'gaps':gaps,'outputs':[identity(p) for p in out.glob('*.gz')]}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1)+'\n');print('Evidence',count,'profiles',profile_count,'missing issue/channel pairs',len(gaps),flush=True)


def export_regional(request,root,out):
    out.mkdir(parents=True,exist_ok=False);req=json.loads(request.read_text());byday=collections.defaultdict(set)
    for c in req['cases']:byday[c['issue_date']].add(c['city'])
    sources=[];coverage=[];rows_out=0
    with gzip.open(out/'regional_city_rows.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as f:
        for model in ('cmaq','naqp','wrf'):
            readme=root/model/'readme.md'
            if readme.is_file():(out/f'{model}_source_readme.txt').write_text(readme.read_text(errors='replace'))
            for frequency in ('day','hour'):
                for domain in ('d02','d03'):
                    selected_total=0
                    for day,cities in sorted(byday.items()):
                        ref=date.fromisoformat(day)-timedelta(days=1);p=root/model/frequency/f'{model}_{domain}_{ref:%Y%m%d}20.csv'
                        entry={'model':model,'frequency':frequency,'domain':domain,'issue_date':day,'path':str(p),'exists':p.is_file()}
                        if not p.is_file():coverage.append(entry);continue
                        selection=[];citymatches=collections.Counter()
                        with p.open(encoding='utf-8-sig',newline='') as h:
                            reader=csv.DictReader(h);fields=reader.fieldnames
                            if not fields or 'cityname' not in fields:entry['failure']='cityname_column_missing';coverage.append(entry);continue
                            for row in reader:
                                city=row.get('cityname','').removesuffix('市')
                                if city in cities:selection.append(row);citymatches[city]+=1
                        entry.update(selected_rows=len(selection),cities_with_rows=dict(citymatches),missing_requested_cities=sorted(cities-citymatches.keys()))
                        if selection:
                            ident=identity(p);sources.append(ident);entry['source_sha256']=ident['sha256'];entry['fields']=fields
                            for row in selection:
                                write(f,{'model':model,'frequency':frequency,'domain':domain,'issue_date':day,'cycle_bjt':f'{ref}T20:00:00+08:00','verified_available_at':None,'units_note':'raw CSV strings; consult preserved source README; no inferred conversions','source_sha256':ident['sha256'],'row':row})
                            rows_out+=len(selection);selected_total+=len(selection)
                        coverage.append(entry)
                    print('Regional',model,frequency,domain,'selected rows',selected_total,flush=True)
    manifest={'request':identity(request),'extractor':identity(Path(__file__)),'records':rows_out,'sources':sources,'coverage':coverage,'raw_files_transferred':False,'publication_verified':False,'outputs':[identity(p) for p in out.glob('*.gz')]}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1)+'\n')


def export_expert(root,out):
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted(root.glob('解压_20260807对话校验原文*/*/*verified.json'))+sorted(root.glob('解压_20260807语料导出*/*/*/*sft.jsonl'))
    sources=[];count=0;bykind=collections.Counter()
    with gzip.open(out/'expert_text_and_distillations.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as f:
        for p in paths:
            ident=identity(p);sources.append(ident);m=re.search(r'(2025)年(\d+)月(\d+)日',p.name)
            meeting_date=f'{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}' if m else None
            kind='transcript_with_original_and_cleaned_text' if p.suffix=='.json' else 'derived_sft_not_independently_verified'
            values=[json.loads(p.read_text())] if p.suffix=='.json' else [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            for i,data in enumerate(values):
                write(f,{'source_file':p.name,'source_sha256':ident['sha256'],'source_record':i+1,'source_kind':kind,'meeting_date_claim':meeting_date,'date_basis':'filename; preserve any document metadata separately','published_at':None,'source_and_date_independently_verified':False,'not_a_verified_method_card':True,'data':data})
                count+=1;bykind[kind]+=1
    archives=[identity(p) for p in root.glob('20260807*.zip')]
    manifest={'extractor':identity(Path(__file__)),'source_root':str(root),'files':len(sources),'records':count,'record_types':dict(bykind),'sources':sources,'source_archives_not_transferred':archives,'limitations':['Filename meeting dates are not evidence of publication time.','The _verified label is inherited from the corpus producer, not this audit.','Original transcript and cleaned/merged text remain separate fields; SFT is a derived source.','No expert text is attached to forecasting cases or used as future truth.'],'outputs':[identity(p) for p in out.glob('*.gz')]}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1)+'\n');print('Expert',len(sources),'files',count,'records',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('kind',choices=['evidence','regional','expert']);a.add_argument('--request',type=Path);a.add_argument('--source-root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args()
    if x.kind=='evidence':export_evidence(x.request,x.source_root,x.out)
    elif x.kind=='regional':export_regional(x.request,x.source_root,x.out)
    else:export_expert(x.source_root,x.out)
