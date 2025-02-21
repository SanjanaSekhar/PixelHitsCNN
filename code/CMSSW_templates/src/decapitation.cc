#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include "template_utils.h"
#include "PixelGeneric2D.cc"

std::vector<SiPixelTemplateStore> thePixelTemp_; 
std::string templates_dir = "/users/mrogul/Work/NN_CPE/PixelHitsCNN/code/CMSSW_templates/data/";

void loadTemplates(int ID) {
    static bool initialized = false;
    if (!initialized) {
        SiPixelTemplate templ(thePixelTemp_);
        templ.pushfile(ID, thePixelTemp_,templates_dir);
        initialized = true;
    }
}

int getPixMax(int ID, float cota, float cotb) {
    SiPixelTemplate templ(thePixelTemp_);
    templ.interpolate(ID, cota, cotb, -1.f);
    return templ.pixmax();
}

int main() {
    std::string line;
    int ID;
    float cota, cotb;

    std::getline(std::cin, line);
    ID = std::stoi(line);
    loadTemplates(ID);

    while (std::getline(std::cin, line)) {
        std::istringstream stream(line);
        if (!(stream >> ID >> cota >> cotb)) {
            std::cerr << "Invalid input format." << std::endl;
            continue;
        }

        int maxpix = getPixMax(ID, cota, cotb);
        std::cout << maxpix << std::endl;
    }

    return 0;
}
