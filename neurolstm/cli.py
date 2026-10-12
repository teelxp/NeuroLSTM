"""Командный интерфейс; импорт Neuro.py ничего не запускает и не подключается к БД."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .config import load_config
from .data import prepare
from .training import train,evaluate,predict,write_json


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="NeuroLSTM: реализация методов диссертации с паспортом эксперимента")
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo",help="Создать явно синтетические данные, НЕ результаты диссертации")
    demo.add_argument("--output",required=True);demo.add_argument("--rows",type=int,default=480)
    demo.add_argument("--with-semantics",action="store_true")
    validation=sub.add_parser("validate-data",help="Проверить данные, единицы, разбиение и число окон")
    validation.add_argument("--config",required=True);validation.add_argument("--output")
    fitting=sub.add_parser("train",help="Обучить модель без обращения к тестовым метрикам")
    fitting.add_argument("--config",required=True);fitting.add_argument("--output",required=True)
    fitting.add_argument("--resume")
    testing=sub.add_parser("evaluate",help="Независимая оценка сохраненной модели")
    testing.add_argument("--checkpoint",required=True);testing.add_argument("--output",required=True)
    testing.add_argument("--device",default="cpu")
    inference=sub.add_parser("predict",help="Прогноз только из прошлого окна, без будущей телеметрии")
    inference.add_argument("--checkpoint",required=True);inference.add_argument("--csv",required=True)
    inference.add_argument("--output",required=True);inference.add_argument("--device",default="cpu")
    assessment=sub.add_parser("assess",help="Расчет доверия из явно заданных проверок")
    assessment.add_argument("--input",required=True);assessment.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="demo":
            from .demo import make_demo
            print(make_demo(args.output,args.rows,args.with_semantics))
        elif args.command=="validate-data":
            cfg=load_config(args.config);datasets,scaling,audit=prepare(cfg)
            result={"data":audit,"scaling":scaling.to_dict(),"profile":cfg["profile"]}
            if args.output:write_json(args.output,result)
            print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        elif args.command=="train":
            result=train(load_config(args.config),args.output,args.resume)
            print(f"Обучение завершено. Фактическое время: {result['training_seconds']:.3f} с. Test не использовался для выбора.")
        elif args.command=="evaluate":
            result=evaluate(args.checkpoint,args.output,args.device)
            print(json.dumps(result["horizons"],ensure_ascii=False,indent=2))
        elif args.command=="predict":
            result=predict(args.checkpoint,args.csv,args.output,args.device)
            print(f"Сохранено {len(result)} прогнозов. Физическая проверка будущего режима требует независимого сценария.")
        elif args.command=="assess":
            from .decision import consistency_assessment
            payload=json.loads(Path(args.input).read_text(encoding="utf-8"))
            write_json(args.output,consistency_assessment(**payload))
        return 0
    except (ValueError,RuntimeError,OSError,FloatingPointError,KeyError) as exc:
        # DSN не печатается; psycopg-ошибки могут содержать реквизиты хоста,
        # поэтому для БД предпочтительно запускать validate-data в приватной среде.
        print(f"Ошибка: {exc}",file=sys.stderr)
        return 2
