root_flags = `root-config --cflags --ldflags --libs --glibs`
compiler_flags = -std=c++11 -Wall

boostpath = /users/mrogul/Work/NN_CPE/PixelHitsCNN/code/CMSSW_templates/dependencies/boost/include
vdtpath = /users/mrogul/Work/NN_CPE/PixelHitsCNN/code/CMSSW_templates/dependencies/vdt/include

includes = -I. -I../cmssw_code/ -I$(boostpath) -I$(vdtpath) -I../src/
DEPS = template_utils.h ranlux.c ../cmssw_code/*
ODIR = ../bin/
BOOST_FLAG = -DSI_PIXEL_TEMPLATE_USE_BOOST

all: decapitation

decapitation: decapitation.cc $(DEPS)
	c++ $(BOOST_FLAG) $(compiler_flags) decapitation.cc -o $(ODIR)/decapitation $(root_flags) $(includes)

clean:
	rm -f $(ODIR)/*

.PHONY: all clean
