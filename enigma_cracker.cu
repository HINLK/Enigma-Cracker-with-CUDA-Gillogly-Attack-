#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>

// ==================== 常量定义 ====================
#define ALPHABET_SIZE 26
#define ROTOR_COUNT 3
#define MAX_PLUGS 10
#define MAX_CANDIDATES 1000

// ==================== 主机端转子接线（供CPU使用） ====================
static const int h_rotor_wiring[ROTOR_COUNT][ALPHABET_SIZE] = {
    // 转子 I: EKMFLGDQVZNTOWYHXUSPAIBRCJ
    {4, 10, 12, 5, 11, 6, 3, 16, 21, 25, 13, 19, 14, 22, 24, 7, 23, 20, 18, 15, 0, 8, 1, 17, 2, 9},
    // 转子 II: AJDKSIRUXBLHWTMCQGZNPYFVOE
    {0, 9, 3, 10, 18, 8, 17, 20, 23, 1, 11, 7, 22, 19, 12, 2, 16, 6, 25, 13, 15, 24, 5, 21, 14, 4},
    // 转子 III: BDFHJLCPRTXVZNYEIWGAKMUSQO
    {1, 3, 5, 7, 9, 11, 2, 15, 17, 19, 23, 21, 25, 13, 24, 4, 8, 22, 6, 0, 10, 12, 20, 18, 16, 14}
};
static const int h_reflector[ALPHABET_SIZE] = {
    24, 17, 20, 7, 16, 18, 11, 3, 15, 23, 13, 6, 14, 10, 12, 8, 4, 1, 5, 25, 2, 22, 21, 9, 0, 19
};
static const int h_notch_positions[ROTOR_COUNT] = {16, 4, 21};  // Q, E, V

// ==================== 设备常量（转子接线，反射器等） ====================
__constant__ int d_rotor_wiring[ROTOR_COUNT][ALPHABET_SIZE];
__constant__ int d_reflector[ALPHABET_SIZE];
__constant__ int d_notch_positions[ROTOR_COUNT];

// 德语字母频率（用于计算重合指数）
__constant__ float d_german_freq[ALPHABET_SIZE] = {
    0.0651, 0.0189, 0.0306, 0.0508, 0.1740, 0.0166,
    0.0301, 0.0476, 0.0755, 0.0027, 0.0121, 0.0344,
    0.0253, 0.0978, 0.0251, 0.0079, 0.0002, 0.0700,
    0.0727, 0.0615, 0.0435, 0.0067, 0.0189, 0.0003,
    0.0004, 0.0113
};

// ==================== 设备函数：Enigma核心操作 ====================
__device__ char enigma_process_char(char input,
                                     int rotor_pos[ROTOR_COUNT],
                                     int rotor_order[ROTOR_COUNT],
                                     int plugboard[ALPHABET_SIZE]) {
    int c = input - 'A';

    // 正向插线板
    c = plugboard[c];

    // 正向转子
    int current = c;
    for (int i = 0; i < ROTOR_COUNT; i++) {
        int rotor_idx = rotor_order[i];
        int offset = rotor_pos[i];
        int pos = (current + offset) % ALPHABET_SIZE;
        current = (d_rotor_wiring[rotor_idx][pos] - offset + ALPHABET_SIZE) % ALPHABET_SIZE;
    }

    // 反射器
    current = d_reflector[current];

    // 反向转子
    for (int i = ROTOR_COUNT - 1; i >= 0; i--) {
        int rotor_idx = rotor_order[i];
        int offset = rotor_pos[i];
        // 反向查找
        int target = (current + offset) % ALPHABET_SIZE;
        for (int j = 0; j < ALPHABET_SIZE; j++) {
            int mapped = (d_rotor_wiring[rotor_idx][j] - offset + ALPHABET_SIZE) % ALPHABET_SIZE;
            if (mapped == target) {
                current = (j - offset + ALPHABET_SIZE) % ALPHABET_SIZE;
                break;
            }
        }
    }

    // 反向插线板
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (plugboard[i] == current) {
            current = i;
            break;
        }
    }

    return (char)(current + 'A');
}

__device__ void update_rotors(int rotor_pos[ROTOR_COUNT]) {
    rotor_pos[0] = (rotor_pos[0] + 1) % ALPHABET_SIZE;
    if (rotor_pos[0] == d_notch_positions[0]) {
        rotor_pos[1] = (rotor_pos[1] + 1) % ALPHABET_SIZE;
        if (rotor_pos[1] == d_notch_positions[1]) {
            rotor_pos[2] = (rotor_pos[2] + 1) % ALPHABET_SIZE;
        }
    }
}

__device__ void decrypt_message(const char* ciphertext,
                                 char* plaintext,
                                 int length,
                                 int rotor_pos[ROTOR_COUNT],
                                 int rotor_order[ROTOR_COUNT],
                                 int plugboard[ALPHABET_SIZE]) {
    int current_pos[ROTOR_COUNT];
    for (int i = 0; i < ROTOR_COUNT; i++) current_pos[i] = rotor_pos[i];

    for (int i = 0; i < length; i++) {
        plaintext[i] = enigma_process_char(ciphertext[i], current_pos, rotor_order, plugboard);
        update_rotors(current_pos);
    }
}

__device__ float calculate_ic(const char* text, int length) {
    if (length < 2) return 0.0f;
    int freq[ALPHABET_SIZE] = {0};
    for (int i = 0; i < length; i++) {
        if (text[i] >= 'A' && text[i] <= 'Z')
            freq[text[i] - 'A']++;
    }
    float ic = 0.0f;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        ic += freq[i] * (freq[i] - 1);
    ic /= (length * (length - 1));
    return ic;
}

// ==================== 内核：搜索所有转子设置 ====================
__global__ void search_rotor_settings_kernel(
    const char* d_ciphertext,
    int cipher_len,
    float* d_ic_scores,
    int* d_rotor_orders,
    int* d_rotor_positions,
    int* d_result_count) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_settings = 6 * 26 * 26 * 26;  // 105,456
    if (idx >= total_settings) return;

    // 解码转子顺序和初始位置
    int order_idx = idx / (26 * 26 * 26);
    int pos_idx = idx % (26 * 26 * 26);
    int pos1 = pos_idx / (26 * 26);
    int pos2 = (pos_idx / 26) % 26;
    int pos3 = pos_idx % 26;

    int rotor_order[ROTOR_COUNT];
    switch (order_idx) {
        case 0: rotor_order[0]=0; rotor_order[1]=1; rotor_order[2]=2; break;
        case 1: rotor_order[0]=0; rotor_order[1]=2; rotor_order[2]=1; break;
        case 2: rotor_order[0]=1; rotor_order[1]=0; rotor_order[2]=2; break;
        case 3: rotor_order[0]=1; rotor_order[1]=2; rotor_order[2]=0; break;
        case 4: rotor_order[0]=2; rotor_order[1]=0; rotor_order[2]=1; break;
        case 5: rotor_order[0]=2; rotor_order[1]=1; rotor_order[2]=0; break;
    }

    int rotor_pos[ROTOR_COUNT] = {pos1, pos2, pos3};

    // 初始插线板（恒等映射）
    int plugboard[ALPHABET_SIZE];
    for (int i = 0; i < ALPHABET_SIZE; i++) plugboard[i] = i;

    // 使用共享内存存储解密文本
    __shared__ char shared_plain[256];
    char* plaintext = shared_plain;

    int decrypt_len = min(cipher_len, 200);
    decrypt_message(d_ciphertext, plaintext, decrypt_len,
                    rotor_pos, rotor_order, plugboard);

    float ic = calculate_ic(plaintext, decrypt_len);

    // 保存结果
    int pos = atomicAdd(d_result_count, 1);
    if (pos < MAX_CANDIDATES) {
        d_ic_scores[pos] = ic;
        d_rotor_orders[pos] = order_idx;
        d_rotor_positions[pos * 3] = pos1;
        d_rotor_positions[pos * 3 + 1] = pos2;
        d_rotor_positions[pos * 3 + 2] = pos3;
    }
}

// ==================== CPU辅助函数（用于第二阶段） ====================

// CPU版Enigma模拟（供爬山算法使用）
void cpu_enigma_decrypt(const char* ciphertext, char* plaintext, int len,
                        int rotor_order[3], int rotor_pos[3], int plugboard[26]) {
    int pos[3] = {rotor_pos[0], rotor_pos[1], rotor_pos[2]};
    for (int i = 0; i < len; i++) {
        int c = ciphertext[i] - 'A';

        // 正向插线板
        c = plugboard[c];

        // 正向转子
        int current = c;
        for (int r = 0; r < 3; r++) {
            int rotor_idx = rotor_order[r];
            int offset = pos[r];
            int inp = (current + offset) % 26;
            current = (h_rotor_wiring[rotor_idx][inp] - offset + 26) % 26;
        }

        // 反射器
        current = h_reflector[current];

        // 反向转子
        for (int r = 2; r >= 0; r--) {
            int rotor_idx = rotor_order[r];
            int offset = pos[r];
            int target = (current + offset) % 26;
            for (int j = 0; j < 26; j++) {
                int mapped = (h_rotor_wiring[rotor_idx][j] - offset + 26) % 26;
                if (mapped == target) {
                    current = (j - offset + 26) % 26;
                    break;
                }
            }
        }

        // 反向插线板
        for (int j = 0; j < 26; j++) {
            if (plugboard[j] == current) {
                current = j;
                break;
            }
        }

        plaintext[i] = current + 'A';

        // 更新转子
        pos[0] = (pos[0] + 1) % 26;
        if (pos[0] == h_notch_positions[0]) {
            pos[1] = (pos[1] + 1) % 26;
            if (pos[1] == h_notch_positions[1]) {
                pos[2] = (pos[2] + 1) % 26;
            }
        }
    }
}

// 计算三元组评分（使用简单频率表）
float trigram_score(const char* text, int len) {
    // 简化的德语常用三元组（仅示例，实际应使用完整频率）
    static const char* common_trigrams[] = {"DER","DIE","UND","EIN","CHT","DEN","INE","UNG","TEN","SCH"};
    static const float scores[] = {10,9,8,7,6,5,4,3,2,1};
    int n = sizeof(common_trigrams)/sizeof(common_trigrams[0]);

    float score = 0.0f;
    for (int i = 0; i < len-2; i++) {
        char tri[4] = {text[i], text[i+1], text[i+2], 0};
        for (int j = 0; j < n; j++) {
            if (strncmp(tri, common_trigrams[j], 3) == 0) {
                score += scores[j];
                break;
            }
        }
    }
    return score;
}

// 爬山算法寻找插线板
void find_plugboard(const char* ciphertext, int len,
                    int rotor_order[3], int rotor_pos[3]) {
    int plugboard[26];
    for (int i = 0; i < 26; i++) plugboard[i] = i;

    char* plaintext = new char[len+1];
    plaintext[len] = 0;

    cpu_enigma_decrypt(ciphertext, plaintext, len, rotor_order, rotor_pos, plugboard);
    float best_score = trigram_score(plaintext, len);
    printf("Initial trigram score: %.2f\n", best_score);

    bool improved = true;
    int iter = 0;
    while (improved && iter < 1000) {
        improved = false;
        for (int a = 0; a < 26; a++) {
            for (int b = a+1; b < 26; b++) {
                if (plugboard[a] != a || plugboard[b] != b) continue;

                // 尝试交换
                plugboard[a] = b;
                plugboard[b] = a;
                cpu_enigma_decrypt(ciphertext, plaintext, len, rotor_order, rotor_pos, plugboard);
                float score = trigram_score(plaintext, len);

                if (score > best_score) {
                    best_score = score;
                    improved = true;
                    printf("  iter %d: added %c-%c, score=%.2f\n",
                           iter, 'A'+a, 'A'+b, best_score);
                } else {
                    // 恢复
                    plugboard[a] = a;
                    plugboard[b] = b;
                }
            }
        }
        iter++;
    }

    printf("\nFinal plugboard:\n");
    for (int i = 0; i < 26; i++) {
        if (plugboard[i] != i && i < plugboard[i])
            printf("  %c <-> %c\n", 'A'+i, 'A'+plugboard[i]);
    }
    printf("\nDecrypted message:\n%s\n", plaintext);
    delete[] plaintext;
}

// ==================== 主函数 ====================
int main(int argc, char** argv) {
    // 初始化设备常量（将主机数组拷贝到GPU常量内存）
    cudaMemcpyToSymbol(d_rotor_wiring, h_rotor_wiring, sizeof(h_rotor_wiring));
    cudaMemcpyToSymbol(d_reflector, h_reflector, sizeof(h_reflector));
    cudaMemcpyToSymbol(d_notch_positions, h_notch_positions, sizeof(h_notch_positions));

    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device %d: %s (Compute %d.%d, %d SMs)\n",
               i, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    }
    if (deviceCount == 0) {
        printf("No CUDA device found. Exiting.\n");
        return 1;
    }

    // 选择第一个GPU（单卡）
    cudaSetDevice(0);

    // ===== 读取密文 =====
    // 方法1：直接在代码中指定（请替换为您实际的密文）
    const char* ciphertext = "YOUR_ENIGMA_CIPHERTEXT_HERE";  // <-- 请修改这里
    int len = strlen(ciphertext);

    // 方法2：从文件读取（推荐）
    /*
    FILE* f = fopen("cipher.txt", "r");
    if (!f) { printf("Cannot open cipher.txt\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* ciphertext = (char*)malloc(fsize+1);
    fread(ciphertext, 1, fsize, f);
    fclose(f);
    ciphertext[fsize] = 0;
    int len = fsize;
    // 去除换行符等（可选）
    */

    printf("\nCiphertext (%d chars): %s\n\n", len, ciphertext);

    // ===== 第一阶段：GPU搜索转子设置 =====
    printf("=== Phase 1: GPU search for rotor settings ===\n");

    char* d_ciphertext;
    float* d_ic_scores;
    int* d_rotor_orders;
    int* d_rotor_positions;
    int* d_result_count;

    cudaMalloc(&d_ciphertext, len);
    cudaMalloc(&d_ic_scores, MAX_CANDIDATES * sizeof(float));
    cudaMalloc(&d_rotor_orders, MAX_CANDIDATES * sizeof(int));
    cudaMalloc(&d_rotor_positions, MAX_CANDIDATES * 3 * sizeof(int));
    cudaMalloc(&d_result_count, sizeof(int));

    cudaMemcpy(d_ciphertext, ciphertext, len, cudaMemcpyHostToDevice);
    cudaMemset(d_result_count, 0, sizeof(int));

    int total_settings = 6 * 26 * 26 * 26;
    int threads = 256;
    int blocks = (total_settings + threads - 1) / threads;

    search_rotor_settings_kernel<<<blocks, threads>>>(
        d_ciphertext, len,
        d_ic_scores,
        d_rotor_orders,
        d_rotor_positions,
        d_result_count
    );

    cudaDeviceSynchronize();

    int h_result_count;
    cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU returned %d candidate rotor settings\n", h_result_count);

    if (h_result_count == 0) {
        printf("No candidates found. Exiting.\n");
        return 0;
    }

    float* h_ic_scores = new float[h_result_count];
    int* h_rotor_orders = new int[h_result_count];
    int* h_rotor_positions = new int[h_result_count * 3];

    cudaMemcpy(h_ic_scores, d_ic_scores, h_result_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rotor_orders, d_rotor_orders, h_result_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rotor_positions, d_rotor_positions, h_result_count * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // 按重合指数降序排序（简单选择排序）
    for (int i = 0; i < h_result_count-1; i++) {
        for (int j = i+1; j < h_result_count; j++) {
            if (h_ic_scores[j] > h_ic_scores[i]) {
                std::swap(h_ic_scores[i], h_ic_scores[j]);
                std::swap(h_rotor_orders[i], h_rotor_orders[j]);
                std::swap(h_rotor_positions[i*3], h_rotor_positions[j*3]);
                std::swap(h_rotor_positions[i*3+1], h_rotor_positions[j*3+1]);
                std::swap(h_rotor_positions[i*3+2], h_rotor_positions[j*3+2]);
            }
        }
    }

    printf("\nTop 5 rotor candidates:\n");
    for (int i = 0; i < min(5, h_result_count); i++) {
        printf("  %d: IC=%.4f order=%d pos=(%d,%d,%d)\n",
               i, h_ic_scores[i], h_rotor_orders[i],
               h_rotor_positions[i*3], h_rotor_positions[i*3+1], h_rotor_positions[i*3+2]);
    }

    // ===== 第二阶段：对每个候选执行插线板爬山 =====
    printf("\n=== Phase 2: CPU hill-climbing for plugboard ===\n");
    for (int cand = 0; cand < min(3, h_result_count); cand++) {
        if (h_ic_scores[cand] < 0.07) continue;  // 忽略太低IC的

        int order_idx = h_rotor_orders[cand];
        int rotor_order[3];
        switch (order_idx) {
            case 0: rotor_order[0]=0; rotor_order[1]=1; rotor_order[2]=2; break;
            case 1: rotor_order[0]=0; rotor_order[1]=2; rotor_order[2]=1; break;
            case 2: rotor_order[0]=1; rotor_order[1]=0; rotor_order[2]=2; break;
            case 3: rotor_order[0]=1; rotor_order[1]=2; rotor_order[2]=0; break;
            case 4: rotor_order[0]=2; rotor_order[1]=0; rotor_order[2]=1; break;
            case 5: rotor_order[0]=2; rotor_order[1]=1; rotor_order[2]=0; break;
        }
        int rotor_pos[3] = {h_rotor_positions[cand*3],
                            h_rotor_positions[cand*3+1],
                            h_rotor_positions[cand*3+2]};

        printf("\n--- Processing candidate %d ---\n", cand);
        find_plugboard(ciphertext, len, rotor_order, rotor_pos);
    }

    // 清理
    delete[] h_ic_scores;
    delete[] h_rotor_orders;
    delete[] h_rotor_positions;
    cudaFree(d_ciphertext);
    cudaFree(d_ic_scores);
    cudaFree(d_rotor_orders);
    cudaFree(d_rotor_positions);
    cudaFree(d_result_count);

    // 如果从文件读取了密文，释放内存
    // free(ciphertext);

    return 0;
}
