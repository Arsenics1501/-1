#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#ifdef SIMD
#include <arm_neon.h>
#endif

#include "common.h"
#include "compute.h"

void zero_z() {
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            Z[i][j] = 0;
        }
    }
}

void compute_row_major_mnk() {
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_mkn() {
     zero_z();
    for (int i = 0; i != m; ++i) {
        for (int l = 0; l != k; ++l) {
            for (int j = 0; j != n; ++j) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_kmn() {
     zero_z();
    for (int l = 0; l != k; ++l) {
        for (int i = 0; i != m; ++i) {
            for (int j = 0; j != n; ++j) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_nmk() {
     zero_z();
    for (int j = 0; j != n; ++j) {
        for (int i = 0; i != m; ++i) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_nkm() {
     zero_z();
    for (int j = 0; j != n; ++j) {
        for (int l = 0; l != k; ++l) {
            for (int i = 0; i != m; ++i) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_knm() {
     zero_z();
    for (int l = 0; l != k; ++l) {
        for (int j = 0; j != n; ++j) {
            for (int i = 0; i != m; ++i) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_y_transpose_mnk() {
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * YP[j][l];
            }
        }
    }
    // TODO: task 2
}

void compute_row_major_mnkkmn_b32() {
        zero_z();
    for (int ii = 0; ii != m/32; ++ii) {
        for (int jj = 0; jj != n/32; ++jj) {
            for (int ll = 0; ll != k/32; ++ll) {
                for(int l = 0; l != 32; ++l) {
                    for(int i = 0; i != 32; ++i){
                        for(int j = 0; j != 32; ++j){
                Z[ii*32+i][jj*32+j] += X[ii*32+i][ll*32+l] * Y[ll*32+l][jj*32+j];
                        }
                    }
                }
            }
        }
    }
    // TODO: task 2
}

void compute_row_major_mnk_lu2() {
        zero_z();
    for (int i = 0; i != m; i += 2) {
        for (int j = 0; j != n; j += 2) {
            for (int l = 0; l != k; l +=2) {
                Z[i][j] += X[i][l] * Y[l][j];
                Z[i+1][j] += X[i+1][l] * Y[l][j];
                Z[i][j+1] += X[i][l] * Y[l][j+1];
                Z[i][j] += X[i][l+1] * Y[l+1][j];
                Z[i+1][j+1] += X[i+1][l] * Y[l][j+1];
                Z[i+1][j] += X[i+1][l+1] * Y[l+1][j];
                Z[i][j+1] += X[i][l+1] * Y[l+1][j+1];
                Z[i+1][j+1] += X[i+1][l+1] * Y[l+1][j+1];
            }
        }
    }
    // TODO: task 2
}

void compute_simd() {
#ifdef SIMD
    matrix_mul(X,Y,Z,m,n,k){
        for (int i = 0; i != m; ++i) {
            for (int j = 0; j != n; ++j) {
                for (int l = 0; l != k; ++l) {
                    Z[i][j] += X[i][l] * Y[l][j];
                }
            }
        }
    }
    // TODO: task 3
#endif
}

uint64_t elapsed(const struct timespec start, const struct timespec end) {
    struct timespec result;
    result.tv_sec = end.tv_sec - start.tv_sec;
    result.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (result.tv_nsec < 0) {
        --result.tv_sec;
        result.tv_nsec += SEC;
    }
    uint64_t res = result.tv_sec * SEC + result.tv_nsec;
    return res;

}

uint64_t compute() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    switch (COMPUTE_SELECT) {
        case COMPUTE_ROW_MAJOR_MNK:
            //printf("COMPUTE_ROW_MAJOR_MNK\n");
            compute_row_major_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MKN:
            //printf("COMPUTE_ROW_MAJOR_MKN\n");
            compute_row_major_mkn();
            break;
        case COMPUTE_ROW_MAJOR_KMN:
            //printf("COMPUTE_ROW_MAJOR_KMN\n");
            compute_row_major_kmn();
            break;
        case COMPUTE_ROW_MAJOR_NMK:
            //printf("COMPUTE_ROW_MAJOR_NMK\n");
            compute_row_major_nmk();
            break;
        case COMPUTE_ROW_MAJOR_NKM:
            //printf("COMPUTE_ROW_MAJOR_NKM\n");
            compute_row_major_nkm();
            break;
        case COMPUTE_ROW_MAJOR_KNM:
            //printf("COMPUTE_ROW_MAJOR_KNM\n");
            compute_row_major_knm();
            break;
        case COMPUTE_Y_TRANSPOSE_MNK:
            //printf("COMPUTE_Y_TRANSPOSE_MNK\n");
            compute_y_transpose_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MNKKMN_B32:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B32\n");
            compute_row_major_mnkkmn_b32();
            break;
        case COMPUTE_ROW_MAJOR_MNK_LU2:
            //printf("COMPUTE_ROW_MAJOR_MNK_LU2\n");
            compute_row_major_mnk_lu2();
            break;
        case COMPUTE_SIMD:
            //printf("COMPUTE_SIMD\n");
            compute_simd();
            break;
        default:
            printf("Unreachable!");
            return 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed(start, end);
}

