# conda activate doge
# conda activate dm2f
# pip install -r requirements.txt

python preprocess/object_detection.py
python preprocess/preprocess.py


python train.py
python evaluate.py


python train_video.py
python evaluate_video.py


python train_transformer.py --heads 3 --scaledim 4 --depth 4 --device 'cuda:1'
python train_transformer.py --heads 6 --scaledim 4 --depth 4 --device 'cuda:1'
python train_transformer.py --heads 3 --scaledim 4 --depth 8 --device 'cuda:1'
python train_transformer.py --heads 6 --scaledim 4 --depth 12 --device 'cuda:1'
python train_transformer.py --heads 12 --scaledim 8 --depth 12 --device 'cuda:2'