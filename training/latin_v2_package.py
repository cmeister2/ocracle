#!/usr/bin/env python3
"""Pin the completed V2 research artifact without registering language profiles."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    model=ROOT/'models/ocracle-mtg-latin-v2.rten';report_path=ROOT/'training/latin-v2-report.json'
    report=json.loads(report_path.read_text())
    validation=json.loads((ROOT/'training/latin-v2-reproduction-validation.json').read_text())
    if validation['manifest_sha256']!=report['manifest_sha256']:raise RuntimeError('Report and replay refer to different corpora')
    configurations={language:f'models/latin-v2-profiles/{language}.json' for language in report['languages']}
    manifest={'schema':1,'path':str(model.relative_to(ROOT)),'sha256':sha(model),'bytes':model.stat().st_size,'model_name':report['model_name'],'languages':report['languages'],'license':'CC-BY-SA-4.0','status':'Experimental CRNN research artifact; not registered as a ready language profile','configurations':configurations,'configuration_sha256':{language:sha(ROOT/path) for language,path in configurations.items()},'source_checkpoint':report['source_checkpoint'],'training_report':str(report_path.relative_to(ROOT)),'training_report_sha256':sha(report_path),'dataset_manifest_sha256':report['manifest_sha256'],'data_replay':'training/latin-v2-reproduction-validation.json','data_coverage':'training/latin-v2-corpus/coverage.json','physical_review':'training/latin-v2-corpus/titles-reviewed.json','training_steps':report['steps'],'validation_selected_step':report['selected_step'],'batch_norm_running_statistics_frozen':True,'trainable_backbone':report['trainable_backbone'],'limitations':report['limitations']+['Bounded physical test set; Italian has one title. No universal font, language or degradation guarantee.','Opt is excluded from V2 updates but was present in V1 initialization training.']}
    evaluation=ROOT/'training/latin-eval/results-v2.json'
    if evaluation.exists() and json.loads(evaluation.read_text())['recognition_model_sha256']==sha(model):
        manifest['independent_physical_evaluation']=str(evaluation.relative_to(ROOT));manifest['independent_physical_evaluation_sha256']=sha(evaluation)
        manifest['limitations']=[text.replace('unregistered pending independent physical evaluation.', 'unregistered because independent physical evaluation still shows substantial accent errors.') for text in manifest['limitations']]
    gallery=ROOT/'training/latin-v2-english-gallery.json'
    if gallery.exists() and json.loads(gallery.read_text())['recognition_model_sha256']==sha(model):
        manifest['english_font_evaluation']=str(gallery.relative_to(ROOT));manifest['english_font_evaluation_sha256']=sha(gallery)
    manifest['documentation']='docs/training/latin-v2.md'
    manifest['export_checks']={language:f'training/latin-v2-export-validation-{language}.json' for language in report['languages'] if (ROOT/f'training/latin-v2-export-validation-{language}.json').exists() and json.loads((ROOT/f'training/latin-v2-export-validation-{language}.json').read_text())['model_sha256']==sha(model)}
    manifest['limitations'][-2]=f"Only {report['v2_test']['unique_lines']} new physical test title crops; Italian has one. No universal font, language or degradation guarantee."
    (ROOT/'models/latin-v2-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
