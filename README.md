# projet_CHPS_801

```
module load opencv/
module load gcc/ (une version 4)
```

# partie openmp task
```
cd src
make clean
make
./opencv_test.pgr votreimage
```
gaussSeidelTaks.cpp => version openmp task de gauss seidel et découpage en deux (divide to conqueer) fonction AddGaussSeidelTask
gaussSeidel.cpp => version naîf et version parcours en diagonal - fonction AddGaussSeidel_wave et AddGaussSeidel