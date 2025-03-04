#ifndef HEADER_H
#define HEADER_H

#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <windows.h>
#include <mmsystem.h>
#include <vector>
#include <fstream>
#include <cstdint>
#include <stdio.h>
#include <conio.h> // For kbhit() and getch()
#include <iomanip>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


const int WIDTH    = 238; //952;//238;//1904;//317;//476
const int HEIGHT   = 143; //429;//143;//1072;//177;//268


inline void SetCoursor(short a, short b){
    SetConsoleCursorPosition(
        GetStdHandle(STD_OUTPUT_HANDLE), COORD {a, b});
}



std::vector<uint8_t> readBMP(const char*, int& , int& );

std::vector<uint8_t> rotateAndRasterizeTexturedCube(float, float, float);

std::vector<uint8_t> fuseVectorsCUDA(const std::vector<uint8_t>& source, const std::vector<uint8_t>& target);

void output_buffer_cuda(const std::vector<uint8_t>&,int);
void test();
#endif // HEADER_H