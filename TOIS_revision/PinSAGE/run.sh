g++ -fPIC -shared -o sklib.so -g -rdynamic -mavx2 -mbmi -mavx512bw -mavx512dq --std=c++17 -O3 -fopenmp sketch.cpp

python model.py data_processed --num-epochs 50 --compress-tech hash --compress-ratio 1 --num-workers 4 --batch-size 256 --device cuda:1 --hidden-dims 64 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee "logs/ideal_$(date +%Y%m%d_%H%M%S).log"
python model.py data_processed --num-epochs 50 --compress-tech cafe --compress-ratio 16 --num-workers 4 --batch-size 256 --device cuda:1 --hidden-dims 64 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee "logs/cafe_16_$(date +%Y%m%d_%H%M%S).log"
python model.py data_processed --num-epochs 50 --compress-tech hash --compress-ratio 16 --num-workers 4 --batch-size 256 --device cuda:1 --hidden-dims 64 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee "logs/hash_128_$(date +%Y%m%d_%H%M%S).log"
