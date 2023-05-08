all: opencv_test.pgr

CC=g++
CFLAGS=-Wall -O3 -fopenmp
LDFLAGS= 
CLANG=clang --gcc-toolchain=/apps/2021/gcc/10.2/ -lstdc++

CFLAGS += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

opencv_test.pgr: obj/opencv_simple.o obj/gaussianNoise.o obj/gaussSeidel.o obj/gaussSeidelTask.o obj/jacobi.o
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)

#llvm_opencv_test.pgr: objllvm/opencv_simple.o objllvm/gaussianNoise.o objllvm/gaussSeidel.o objllvm/gaussSeidelTask.o objllvm/jacobi.o
#	$(CLANG) $(CFLAGS) $^ -o $@ $(LDFLAGS)

#objllvm/%.o: %.cpp
#	$(CLANG) $(CFLAGS) $< -c -o $@

obj/%.o: %.cpp
	$(CC) $(CFLAGS) $< -c -o $@

proper :
	rm -f obj/*.o
#	rm -f objllvm/*.o

clean:
	rm -rf obj/*.o *.pgr
#	rm -rf objllvm/*.o
