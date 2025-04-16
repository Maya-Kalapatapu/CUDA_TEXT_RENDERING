TARGET = cuda_font_renderer
BENCHMARK_TARGET = benchmark_render

CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17 -O2 -I/usr/include/freetype2
LDFLAGS = -lGL -lGLEW -lglfw -lfreetype -lcuda -lcudart

# Main application sources
SRC_MAIN = main.cpp \
           cuda_text.cpp \
           cuda_text_wrapper.cpp \
           font_loader.cpp \
           page_manager.cpp \
           portable_doc.cpp \
           cuda_renderer.cu

OBJ_MAIN = $(SRC_MAIN:.cpp=.o)
OBJ_MAIN := $(OBJ_MAIN:.cu=.o)

# Benchmark-specific sources (no main.cpp!)
SRC_BENCHMARK = benchmark_rendering.cpp \
                cuda_text.cpp \
                cuda_text_wrapper.cpp \
                font_loader.cpp \
                page_manager.cpp \
                portable_doc.cpp \
                cuda_renderer.cu

OBJ_BENCHMARK = $(SRC_BENCHMARK:.cpp=.o)
OBJ_BENCHMARK := $(OBJ_BENCHMARK:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJ_MAIN)
	$(NVCC) -o $@ $^ $(LDFLAGS)

benchmark_render: $(OBJ_BENCHMARK)
	$(CXX) -o $(BENCHMARK_TARGET) $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -x cu -Xcompiler "-fPIC" -c $< -o $@

clean:
	rm -f *.o $(TARGET) $(BENCHMARK_TARGET)
