TARGET = cuda_font_renderer
SRC = $(wildcard *.cpp) $(wildcard *.cu)
OBJ = $(SRC:.cpp=.o)
OBJ := $(OBJ:.cu=.o)

CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17 -O2 -I/usr/include/freetype2
LDFLAGS = -lGL -lGLEW -lglfw -lfreetype -lcuda -lcudart

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -x cu -Xcompiler "-fPIC" -c $< -o $@

clean:
	rm -f *.o $(TARGET)
