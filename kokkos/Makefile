KOKKOS_PATH ?= /home/cchevalier/kokkos_work/kokkos
KOKKOS_DEVICES = "OpenMP"
EXE_NAME = "redBlackJacobi"

SRC = redBlackJacobi.cpp

CC=g++
CFLAGS=-Wall -O3
LDFLAGS=

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Pascal60"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "SKX"
endif

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

CFLAGS += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

opencv_test.pgr: obj/opencv_simple.o obj/gaussianNoise.o
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)

obj/%.o: %.cpp
	$(CC) $(CFLAGS) $< -c -o $@

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE) 

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
