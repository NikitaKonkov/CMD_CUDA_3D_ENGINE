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
std::vector<uint8_t> admin = readBMP("C:/Users/nikit/Desktop/Game Project [Weird Slasher]/Game_0.3v/images/RGB.bmp", w, h);
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

// Kernel to rotate vertices and perform orthographic projection.
__global__ void rotateAndProjectVertices(Vertex* vertices, int numVertices,
    float angleY, float angleX,
    float* transformedVertices) {
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

// Scale for visibility.
rotatedX *= 50.0f;
rotatedY *= 50.0f;

// Screen centering.
transformedVertices[idx * 3]     = rotatedX + WIDTH / 2.0f;
transformedVertices[idx * 3 + 1] = rotatedY + HEIGHT / 2.0f;
transformedVertices[idx * 3 + 2] = rotatedZ;
}

std::vector<uint_fast8_t> rotateAndRasterizeTexturedCube(float angleY, float angleX)
{
    const int numVertices = 8;

    Vertex h_vertices[numVertices] = {
       {-1, -1, -1},
       { 1, -1, -1},
       { 1,  1, -1},
       {-1,  1, -1},
       {-1, -1,  1},
       { 1, -1,  1},
       { 1,  1,  1},
       {-1,  1,  1}
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

    // Launch vertex transformation kernel.
    rotateAndProjectVertices<<<vertexGrid, vertexBlock>>>(d_vertices, numVertices, angleY, angleX, d_transformedVertices);
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

