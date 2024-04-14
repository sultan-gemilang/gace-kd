### Additional Files
Dipake kalau mau ngetes inference aja  
Complete contents for assets/bromelia folder is here:
[Google Drive](https://drive.google.com/drive/folders/15nl2jvvBTT7RTd0xUFealuWfcXme9BhX?usp=sharing)

### Contoh Cara Running
> Dataset sementara kunci dulu pakai oxford_flowers102. distillation.py otomatis ngambil teacher ke generated/oxford_flowers102. 

1. generator.py
    - `--dataset`: nama dataset
    - `--module-name`: preprocessing pakai modul apa, ini sesuaikan sama model yg mau dipakai.
    - `--model-name`: model yang mau di generate
    - `--image-size`: ukuran imagenya mau sekian kali sekian (kalau isi 224, berarti 224 x 224)
    - `--run-num`: angka running. (1, 2, 3... etc.)

    ```
    python3 generator.py --dataset oxford_flowers102 --module-name resnet --model-name ResNet101 --image-size 224 --run-num 1
    ```

2. distillation.py
    - `--module-name`: preprocessing buat student modelnya.
    - `--student-model`: nama student model
    - `--teacher`: nama teacher (teacher harus ada dulu di folder generator/oxford_flowers102)
    - `--run-num`: angka running. (1, 2, 3... etc.)

    ```
    python3 distillation.py --module-name mobilenet_v3 --student-model MobileNetV3Small --teacher ResNet101 --run-num 1
    ```
