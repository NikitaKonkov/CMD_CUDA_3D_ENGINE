#include "../header.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
// Include your header file if needed
//#include "../header.h"

#pragma pack(push, 1)
struct BmpFileHeader {
    uint16_t signature;
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
};

struct BmpInfoHeader {
    uint32_t size;
    int32_t  width;
    int32_t  height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t  xResolution;
    int32_t  yResolution;
    uint32_t nColors;
    uint32_t importantColors;
};
#pragma pack(pop)

std::vector<uint8_t> readBMP(const char* filename, int& outWidth, int& outHeight) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        std::cerr << "Could not open file: " << filename << "\n";
        return {}; // Return an empty vector on failure
    }
    
    BmpFileHeader fileHeader;
    if (fread(&fileHeader, sizeof(fileHeader), 1, file) != 1) {
        std::cerr << "Failed to read BMP file header.\n";
        fclose(file);
        return {};
    }
    
    // Check signature; BMP files have 'BM' (0x4D42) in little-endian.
    if (fileHeader.signature != 0x4D42) {
        std::cerr << "Not a valid BMP file (bad signature).\n";
        fclose(file);
        return {};
    }
    
    BmpInfoHeader infoHeader;
    if (fread(&infoHeader, sizeof(infoHeader), 1, file) != 1) {
        std::cerr << "Failed to read BMP info header.\n";
        fclose(file);
        return {};
    }
    
    if (infoHeader.bitsPerPixel != 24) {
        std::cerr << "Only 24-bit BMP files are supported.\n";
        fclose(file);
        return {};
    }
    
    int width = infoHeader.width;
    int height = std::abs(infoHeader.height);
    outWidth = width;
    outHeight = height;
    
    // Preallocate a vector to hold the fixed-length image data.
    std::vector<uint8_t> pixels(width * height);
    
    // Calculate the size of a row in the BMP file and its padding bytes.
    int rowSize = width * 3;
    int rowPadding = (4 - (rowSize % 4)) % 4;
    
    // Set the file pointer to the beginning of the pixel data.
    fseek(file, fileHeader.dataOffset, SEEK_SET);
    
    // Determine if the image is stored top-down or bottom-up.
    bool topDownInFile = (infoHeader.height < 0);
    
    // Read each row as it appears in the file.
    for (int y = 0; y < height; ++y) {
        // Determine the correct row index in our output image.
        int targetRow = topDownInFile ? y : (height - 1 - y);
        for (int x = 0; x < width; x++) {
            unsigned char bgr[3];
            if (fread(bgr, 3, 1, file) != 1) {
                std::cerr << "Failed to read pixel data.\n";
                fclose(file);
                return {}; // Return an empty vector on failure.
            }
            
            // Convert BGR to a terminal color (using a 216 color scheme)
            uint8_t color = 16
                + (bgr[2] / 43) * 36
                + (bgr[1] / 43) * 6
                + (bgr[0] / 43);
            
            // Place the pixel in the correct position in the vector.
            pixels[targetRow * width + x] = color;
        }
        // Skip the padding bytes at the end of the row.
        fseek(file, rowPadding, SEEK_CUR);
    }
    
    fclose(file);
    return pixels;
}