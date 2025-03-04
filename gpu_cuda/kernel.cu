#include "../header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <windows.h>

// Global variables for BMP reading (assume readBMP is defined properly).
int w, h;
std::vector<uint8_t> admin = readBMP("C:/Users/nikit/Desktop/Game Project [Weird Slasher]/Game_0.3v/images/symb.bmp", w, h);
const int TEX_WIDTH = 128;
const int TEX_HEIGHT = 128;

// Data Structures.
struct Vertex { float x, y, z; };
struct Face { int v[4]; };
struct TexCoord { float u, v; };
struct Edge { int v1, v2; };

// Cube faces: front, back, left, right, top, bottom.
const Face h_CUBE_FACES[6] = {
    { {0, 1, 2, 3} },
    { {4, 5, 6, 7} },
    { {0, 3, 7, 4} },
    { {1, 2, 6, 5} },
    { {3, 2, 6, 7} },
    { {0, 1, 5, 4} }
};

// Copy cube faces to device constant memory.
__constant__ Face d_CUBE_FACES[6];

// Texture coordinates for each face.
__constant__ TexCoord d_faceTexCoords[4] = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {0.0f, 1.0f}
};

// Using constant memory for small textures is acceptable.
__constant__ uint_fast8_t d_texture[TEX_WIDTH * TEX_HEIGHT];

// --- CUDA Kernels ---

// Kernel to rotate vertices and perform orthographic projection.
__global__ void rotateAndProjectVertices(Vertex* vertices, int numVertices,
                                           float angleY, float angleX, 
                                           float translationZ, float* transformedVertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    
    // Rotation about Y-axis.
    float cosY = cosf(angleY), sinY = sinf(angleY);
    float x = vertices[idx].x, y = vertices[idx].y, z = vertices[idx].z;
    float rotatedX = x * cosY - z * sinY;
    float rotatedZ = x * sinY + z * cosY;
    
    // Rotation about X-axis.
    float cosX = cosf(angleX), sinX = sinf(angleX);
    float rotatedY = y * cosX - rotatedZ * sinX;
    rotatedZ = y * sinX + rotatedZ * cosX;
    
    // Translate along Z-axis.
    rotatedZ += translationZ;
    
    // Perspective projection parameters.
    // Here, 'focalLength' can be adjusted to your scene's needs.
    const float focalLength = 300.0f;
    
    // Ensure rotatedZ does not become 0 to avoid divide by zero.
    float perspective = focalLength / (rotatedZ + focalLength); 
    
    // Apply perspective and scaling.
    float screenX = rotatedX * perspective * 50.0f + WIDTH / 2.0f;
    float screenY = rotatedY * perspective * 50.0f + HEIGHT / 2.0f;
    
    // Store transformed vertices.
    transformedVertices[idx * 3]     = screenX;
    transformedVertices[idx * 3 + 1] = screenY;
    transformedVertices[idx * 3 + 2] = rotatedZ;  // Depth value for rasterization.
}

// Rasterization kernel using a 2D configuration.
__global__ void rasterizeTexturedCube(float* transformedVertices, unsigned char* byteMatrix) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= WIDTH || py >= HEIGHT) return;

    float bestDepth = 1e10f;
    uint_fast8_t finalColor = 0; // Background.

    // Pre-load our texture coordinates from constant memory.
    TexCoord tc0 = d_faceTexCoords[0];
    TexCoord tc1 = d_faceTexCoords[1];
    TexCoord tc2 = d_faceTexCoords[2];
    TexCoord tc3 = d_faceTexCoords[3];

    // Iterate over cube faces.
    for (int f = 0; f < 6; f++) {
        Face face = d_CUBE_FACES[f];
        // Fetch vertices from global memory.
        float ax = transformedVertices[face.v[0] * 3];
        float ay = transformedVertices[face.v[0] * 3 + 1];
        float az = transformedVertices[face.v[0] * 3 + 2];
        float bx = transformedVertices[face.v[1] * 3];
        float by = transformedVertices[face.v[1] * 3 + 1];
        float bz = transformedVertices[face.v[1] * 3 + 2];
        float cx = transformedVertices[face.v[2] * 3];
        float cy = transformedVertices[face.v[2] * 3 + 1];
        float cz = transformedVertices[face.v[2] * 3 + 2];
        float dx = transformedVertices[face.v[3] * 3];
        float dy = transformedVertices[face.v[3] * 3 + 1];
        float dz = transformedVertices[face.v[3] * 3 + 2];

        // Process two triangles per face.
        for (int tri = 0; tri < 2; tri++) {
            float vx0, vy0, vz0, u0, v0;
            float vx1, vy1, vz1, u1, v1;
            float vx2, vy2, vz2, u2, v2;
            if (tri == 0) {
                vx0 = ax; vy0 = ay; vz0 = az; u0 = tc0.u; v0 = tc0.v;
                vx1 = bx; vy1 = by; vz1 = bz; u1 = tc1.u; v1 = tc1.v;
                vx2 = cx; vy2 = cy; vz2 = cz; u2 = tc2.u; v2 = tc2.v;
            } else {
                vx0 = ax; vy0 = ay; vz0 = az; u0 = tc0.u; v0 = tc0.v;
                vx1 = cx; vy1 = cy; vz1 = cz; u1 = tc2.u; v1 = tc2.v;
                vx2 = dx; vy2 = dy; vz2 = dz; u2 = tc3.u; v2 = tc3.v;
            }
            // Compute barycentric coordinates.
            float denom = (vy1 - vy2) * (vx0 - vx2) + (vx2 - vx1) * (vy0 - vy2);
            if (fabsf(denom) < 1e-5f) continue;
            float invDenom = 1.0f / denom;
            float pxCenter = px + 0.5f, pyCenter = py + 0.5f;
            float w0 = ((vy1 - vy2) * (pxCenter - vx2) + (vx2 - vx1) * (pyCenter - vy2)) * invDenom;
            float w1 = ((vy2 - vy0) * (pxCenter - vx2) + (vx0 - vx2) * (pyCenter - vy2)) * invDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                float depth = w0 * vz0 + w1 * vz1 + w2 * vz2;
                if (depth < bestDepth) {
                    bestDepth = depth;
                    // Interpolate texture coordinates.
                    float u = w0 * u0 + w1 * u1 + w2 * u2;
                    float v = w0 * v0 + w1 * v1 + w2 * v2;
                    int texX = min(max(int(u * TEX_WIDTH), 0), TEX_WIDTH - 1);
                    int texY = min(max(int(v * TEX_HEIGHT), 0), TEX_HEIGHT - 1);
                    finalColor = d_texture[texY * TEX_WIDTH + texX];
                }
            }
        }
    }
    byteMatrix[py * WIDTH + px] = finalColor;
}

// --- Host Function ---

std::vector<uint_fast8_t> rotateAndRasterizeTexturedCube(float angleY, float angleX, float translationZ)
{
    const int numVertices = 8;
    Vertex h_vertices[numVertices] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}
    };

    // Copy cube faces.
    cudaMemcpyToSymbol(d_CUBE_FACES, h_CUBE_FACES, 6 * sizeof(Face));

    // Create a simple texture using the admin BMP.
    uint_fast8_t h_texture[TEX_WIDTH * TEX_HEIGHT];
    for (int j = 0; j < TEX_WIDTH * TEX_HEIGHT; j++)
        h_texture[j] = admin[j % admin.size()];
    cudaMemcpyToSymbol(d_texture, h_texture, TEX_WIDTH * TEX_HEIGHT * sizeof(uint_fast8_t));

    // Allocate and copy vertex data.
    Vertex* d_vertices;
    cudaMalloc(&d_vertices, numVertices * sizeof(Vertex));
    cudaMemcpy(d_vertices, h_vertices, numVertices * sizeof(Vertex), cudaMemcpyHostToDevice);

    // Allocate memory for transformed vertices.
    float* d_transformedVertices;
    cudaMalloc(&d_transformedVertices, numVertices * 3 * sizeof(float));

    // Allocate memory for the frame (output pixels).
    uint_fast8_t* d_byteMatrix;
    cudaMalloc(&d_byteMatrix, WIDTH * HEIGHT * sizeof(uint_fast8_t));

    // Setup kernel execution parameters.
    dim3 vertexBlock(256);
    dim3 vertexGrid((numVertices + vertexBlock.x - 1) / vertexBlock.x);

    // Launch vertex transformation kernel with translationZ.
    rotateAndProjectVertices<<<vertexGrid, vertexBlock>>>(d_vertices, numVertices, angleY, angleX, translationZ, d_transformedVertices);
    cudaDeviceSynchronize();

    // Clear the frame buffer.
    cudaMemset(d_byteMatrix, 0, WIDTH * HEIGHT * sizeof(uint_fast8_t));

    // Use a 2D grid for rasterization.
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);
    rasterizeTexturedCube<<<gridDim, blockDim>>>(d_transformedVertices, d_byteMatrix);
    cudaDeviceSynchronize();

    // Copy the frame back to host.
    std::vector<uint_fast8_t> h_frame(WIDTH * HEIGHT);
    cudaMemcpy(h_frame.data(), d_byteMatrix, WIDTH * HEIGHT * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost);

    // Free device resources.
    cudaFree(d_byteMatrix);
    cudaFree(d_transformedVertices);
    cudaFree(d_vertices);

    return h_frame;
}



































__global__ void rasterizeTriangle(
    float vx0, float vy0, float vz0, float u0, float v0,
    float vx1, float vy1, float vz1, float u1, float v1,
    float vx2, float vy2, float vz2, float u2, float v2,
    unsigned char* output, int screenWidth, int screenHeight)
{
    // Compute bounding box for the triangle.
    int minX = max(0, (int)floorf(min(vx0, min(vx1, vx2))));
    int minY = max(0, (int)floorf(min(vy0, min(vy1, vy2))));
    int maxX = min(screenWidth-1, (int)ceilf(max(vx0, max(vx1, vx2))));
    int maxY = min(screenHeight-1, (int)ceilf(max(vy0, max(vy1, vy2))));
    
    int x = minX + blockIdx.x * blockDim.x + threadIdx.x;
    int y = minY + blockIdx.y * blockDim.y + threadIdx.y;

    if (x > maxX || y > maxY) return;

    // Convert to barycentric coordinates.
    float denom = (vy1 - vy2) * (vx0 - vx2) + (vx2 - vx1) * (vy0 - vy2);
    if (fabsf(denom) < 1e-5f) return;
    
    float invDenom = 1.0f / denom;
    float px = x + 0.5f;
    float py = y + 0.5f;
    
    float w0 = ((vy1 - vy2) * (px - vx2) + (vx2 - vx1) * (py - vy2)) * invDenom;
    float w1 = ((vy2 - vy0) * (px - vx2) + (vx0 - vx2) * (py - vy2)) * invDenom;
    float w2 = 1.0f - w0 - w1;
    
    if (w0 >= 0 && w1 >= 0 && w2 >= 0)
    {
         // Inside the triangle: interpolate depth and texture coordinates.
         float depth = w0 * vz0 + w1 * vz1 + w2 * vz2;
         // ... texture sampling goes here, with your texture stored in constant or global memory.
         // Assume texture sampling yields a color value.
         unsigned char color = ' '; // texture lookup result
         
         output[y * screenWidth + x] = color;
    }
}













































// Special
__global__ void fuseVectorsKernel(const uint_fast8_t* source, const uint_fast8_t* target, 
    uint_fast8_t* result, size_t size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) {
result[idx] = (target[idx] == 0) ? source[idx] : target[idx];
}
}

std::vector<uint_fast8_t> fuseVectorsCUDA(const std::vector<uint_fast8_t>& source, 
       const std::vector<uint_fast8_t>& target) {
if (source.size() != target.size()) {
throw std::invalid_argument("Vectors must have the same size");
}

size_t size = source.size();
std::vector<uint_fast8_t> result(size);

// Allocate device memory
uint_fast8_t *d_source, *d_target, *d_result;
cudaMalloc(&d_source, size);
cudaMalloc(&d_target, size);
cudaMalloc(&d_result, size);

// Copy data to device
cudaMemcpy(d_source, source.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_target, target.data(), size, cudaMemcpyHostToDevice);

// Launch kernel
int blockSize = 256;
int numBlocks = (size + blockSize - 1) / blockSize;
fuseVectorsKernel<<<numBlocks, blockSize>>>(d_source, d_target, d_result, size);

// Copy result back to host
cudaMemcpy(result.data(), d_result, size, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(d_source);
cudaFree(d_target);
cudaFree(d_result);

return result;
}

#include <stdint.h>

// A simple XORSHIFT PRNG; you may use other algorithms if desired.
__device__ uint32_t xorShift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Modified writeEscapeSequence with device-side PRNG for random characters.
__device__ int writeEscapeSequence(char* buffer, uint_fast8_t color, int repeat_count, uint32_t seed) {
    int pos = 0;

    // Write escape sequence: "\033[48;5;{color}m"
    buffer[pos++] = '\033';
    buffer[pos++] = '[';
    buffer[pos++] = '4';
    buffer[pos++] = '8';
    buffer[pos++] = ';';
    buffer[pos++] = '5';
    buffer[pos++] = ';';

    // Write color number using decimal conversion
    if (color >= 100) {
        buffer[pos++] = '0' + (color / 100);
        buffer[pos++] = '0' + ((color / 10) % 10);
        buffer[pos++] = '0' + (color % 10);
    } else if (color >= 10) {
        buffer[pos++] = '0' + (color / 10);
        buffer[pos++] = '0' + (color % 10);
    } else {
        buffer[pos++] = '0' + color;
    }
    buffer[pos++] = 'm';

    // Initialize the PRNG state with the provided seed.
    uint32_t randState = seed;

    // Write random characters – here we choose from the printable ASCII range (33 to 126).
    for (int i = 0; i < repeat_count; i++) {
        // Generate a random number and convert it to a printable character.
        uint32_t rnd1 = xorShift32(&randState);
        int randomChar1 = 33 + (rnd1 % 94);  // (126 - 33 + 1) = 94 possibilities
        buffer[pos++] = randomChar1;

        // Optionally add a second random character.
        uint32_t rnd2 = xorShift32(&randState);
        int randomChar2 = 33 + (rnd2 % 94);
        buffer[pos++] = randomChar2;
    }
    
    return pos;
}

__global__ void process_line_kernel(const uint_fast8_t* input, 
                                  char* output,
                                  int* output_sizes,
                                  const int width,
                                  const int height,int chr) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (line >= height) return;
    
    int line_start = line * width;
    int output_pos = line * width * 20; // Approximate buffer size per line
    int current_pos = 0;
    
    for (int i = 0; i < width;) {
        uint_fast8_t current_color = input[line_start + i];
        int repeat_count = 1;
        
        // Count consecutive colors within the line
        while (i + repeat_count < width && 
               input[line_start + i + repeat_count] == current_color) {
            repeat_count++;
        }
        
        // Use our custom function instead of sprintf
        current_pos += writeEscapeSequence(
            &output[output_pos + current_pos],
            current_color,
            repeat_count,chr
        );
        
        i += repeat_count;
    }
    
    output_sizes[line] = current_pos;
}
void output_buffer_cuda(const std::vector<uint_fast8_t>& pixel_vector, int chr) {
    if (pixel_vector.size() != WIDTH * HEIGHT) {
        throw std::invalid_argument("Vector must have size of w * h");
    }
    
    // Allocate device memory
    uint_fast8_t* d_input;
    char* d_output;
    int* d_output_sizes;
    
    cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(uint_fast8_t));
    cudaMalloc(&d_output, WIDTH * HEIGHT * 20 * sizeof(char)); // Buffer for output
    cudaMalloc(&d_output_sizes, HEIGHT * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_input, pixel_vector.data(), 
               WIDTH * HEIGHT * sizeof(uint_fast8_t), 
               cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (HEIGHT + threadsPerBlock - 1) / threadsPerBlock;
    
    process_line_kernel<<<blocks, threadsPerBlock>>>(
        d_input, d_output, d_output_sizes, WIDTH, HEIGHT,chr);
    
    // Allocate host memory for results
    std::vector<char> output(WIDTH * HEIGHT * 20);
    std::vector<int> output_sizes(HEIGHT);
    
    // Copy results back
    cudaMemcpy(output.data(), d_output,
               WIDTH * HEIGHT * 20 * sizeof(char),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(output_sizes.data(), d_output_sizes,
               HEIGHT * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // Output results
    for (int i = 0; i < HEIGHT; i++) {
        std::cout.write(&output[i * WIDTH * 20], output_sizes[i]);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_sizes);
}