#include "../header.h"
#include <algorithm>

uint8_t color = 0;
std::vector<uint8_t> fuseVectors(const std::vector<uint8_t>& source, std::vector<uint8_t>& target) {
    // Make sure vectors have the same size
    if (source.size() != target.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    
    // Create a copy of target vector
    std::vector<uint8_t> result = target;
    
    // Iterate through the vectors
    for (size_t i = 0; i < target.size(); i++) {
        // Check if target has 16 at current position
        if (target[i] == 0) {
            result[i] = source[i];
        }
    }
    
    return result;
}

void output_buffer(const std::vector<uint8_t>& pixel_vector) { // CPU based output
    if (pixel_vector.size() != WIDTH * HEIGHT) {
        throw std::invalid_argument("Vector must have size of w * h");
    }
    
    std::stringstream ss;
    size_t n = 0;
    size_t current_line = 0;
    
    while (n < WIDTH * HEIGHT) {
        uint8_t current_color = pixel_vector[n];
        size_t repeat_count = 1;
        
        // Count consecutive same colors, but only within the same line
        while (n + repeat_count < WIDTH * HEIGHT &&
               (n + repeat_count) % WIDTH != 0 &&  // Don't cross line boundaries
               pixel_vector[n + repeat_count] == current_color) {
            repeat_count++;
        }
        
        ss << "\033[38;5;" << static_cast<int>(current_color) << "m";
        for (size_t n = 0; n < repeat_count*2; n++){
            ss << static_cast<char>('A'+(rand()%60));
            }
        n += repeat_count;
    }
    
    std::cout << ss.str();
}

#pragma comment(lib, "user32.lib")
float translationZ = 20.0f;   // Move forward (or backward if negative)
int chr = 0;   // Move right (or left if negative)

float y = 0.000, x = 0.000, s = 0.05;
void handleInput() {
    if (GetAsyncKeyState('W') & 0x8000) {
        x -= s;
    }
    if (GetAsyncKeyState('S') & 0x8000) {
        x += s;
    }
    if (GetAsyncKeyState('A') & 0x8000) {
        y -= s;
    }
    if (GetAsyncKeyState('D') & 0x8000) {
        y += s;
    }
    if (GetAsyncKeyState('Q') & 0x8000) {
        s+=0.05;
    }
    if (GetAsyncKeyState('E') & 0x8000) {
        s-=0.05;
    }
    if (GetAsyncKeyState(VK_UP) & 0x8000) {
        translationZ += 10;
    }
    if (GetAsyncKeyState(VK_DOWN) & 0x8000) {
        translationZ -= 10;
    }
    if (GetAsyncKeyState(VK_LEFT) & 0x8000) {
        chr -= 1;  // Move left.
    }
    if (GetAsyncKeyState(VK_RIGHT) & 0x8000) {
        chr += 1;  // Move right.
    }

}


void test() {
    int w, h;
    const char* img = "C:/Users/nikit/Desktop/Game Project [Weird Slasher]/Game_0.3v/images/0_25.bmp";
    std::vector<uint8_t> image = readBMP(img, w , h);
    
    Sleep(2000);
    SetCoursor(0, 0);
    output_buffer(image);
    Sleep(2000);
    system("cls");
    
    while (true){
        handleInput();
        std::vector<uint_fast8_t> frame = rotateAndRasterizeTexturedCube(y, x, translationZ);


        SetCoursor(0,0);
        output_buffer_cuda(fuseVectorsCUDA(image,frame),rand());
    }
}

