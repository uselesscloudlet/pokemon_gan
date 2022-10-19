# pokemon_gan

## Порядок запуска среды и обучения модели
1. python -m venv .venv
2. source .venv/bin/activate || .venv/Scripts/activate.bat для Windows
3. pip install -r requirements.txt
4. python download_data.py
5. Запустить ячейки в train.ipynb
6. Лоссы и результат обучения в папке train_output

При желании можно установить параметры модели в файле conf/model_conf.conf. Поскольку модель захардкожена - менять можно не все параметры.
При обучении модели использовалась DCGAN.
