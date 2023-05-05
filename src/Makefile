all: opencv_test.pgr

CC=g++
CFLAGS=-Wall -O3 -fopenmp
LDFLAGS=

CFLAGS += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

opencv_test.pgr: obj/opencv_simple.o obj/gaussianNoise.o obj/gaussSeidel.o obj/gaussSeidelTask.o obj/jacobi.o
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)

obj/%.o: %.cpp
	$(CC) $(CFLAGS) $< -c -o $@

proper :
	rm -f obj/*.o

clean:
	rm -rf obj/*.o *.pgr
