TARGET := post-traces-processing
COMPRESSED := post-traces-processing-compressed

OPTFLAGS=

ifneq ($(DEBUG_TOOL),1)
	OPTFLAGS +=-O3
else
	OPTFLAGS +=-g -O0
endif

# MOD. Begin. ENHANCED TRACES
TRACES_ENHANCED_DIR=../../../traces_enhanced/
TRACES_ENHANCED_OBJ=$(wildcard $(TRACES_ENHANCED_DIR)/obj/*.o)
include ../../../protobuf_deps.mk
PROTOBUF_CXXFLAGS := $(PROTOBUF_INCLUDES)
# MOD. End. ENHANCED TRACES

$(TARGET): post-traces-processing.cpp
	g++ -std=c++11 $(OPTFLAGS) $(PROTOBUF_CXXFLAGS) $(TRACES_ENHANCED_OBJ) -o $@ $^ $(PROTOBUF_LIBS)

$(COMPRESSED): post-traces-processing-compressed.cpp
	g++ -std=c++17 -Wall -Werror $(OPTFLAGS) $(PROTOBUF_CXXFLAGS) $(TRACES_ENHANCED_OBJ) -o $@ $^ -lz $(PROTOBUF_LIBS)

all: $(TARGET) $(COMPRESSED)

run: $(TARGET)
	./$(TARGET)

run-compressed: $(COMPRESSED)
	./$(COMPRESSED)

clean: 
	rm -f $(TARGET) $(COMPRESSED) *.o

