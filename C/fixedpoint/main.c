#define _CRT_SECURE_NO_WARNINGS
#define data_size 48600
#define input_data_width 32
#define input_data_ch 1
//input data(input_data_width, input_data_width)
#define conv1_in_width input_data_width
#define conv1_in_ch input_data_ch
#define conv1_out_width 24
#define conv1_out_ch 256
#define conv1_stride 1
#define conv2_in_width conv1_out_width
#define conv2_in_ch conv1_out_ch
#define conv2_out_ch 256
#define conv2_out_width 8
#define conv2_stride 2
#define kernel_width 9
#define num_primary_caps 2048
#define dim_primary_caps 8
//primarycaps(dim_primary_caps, num_primary_caps)
#define dim_predic_vector 16
#define num_class 5
#define CLIP_8(a) (a > 127? 127:(a < -128 ? -128: a))
#define CLIP_16(a) (a >  32767?  32767:(a < -32768 ? -32768: a))
//prediction_vectors(num_primary_caps, num_class, dim_predic_vector)
#define num_iterations 1
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

char conv1_kernel[conv1_out_ch * conv1_in_ch * kernel_width * kernel_width];           // ch ���� Ȯ���غ�����
char conv2_kernel[conv2_out_ch * conv2_in_ch * kernel_width * kernel_width];
char conv1_output[conv1_out_ch * conv1_out_width * conv1_out_width];
char conv2_output[conv2_out_ch * conv2_out_width * conv2_out_width];
char digits_W[num_primary_caps * num_class * dim_predic_vector * dim_primary_caps];
char digits_bias[num_class * dim_predic_vector];
short u_hat[num_primary_caps * num_class * dim_predic_vector];
char result_v[num_class * dim_predic_vector];

void convolution(char* input, char* kernel, char* output, int input_width,
    int input_ch, int output_width, int output_ch, int stride) {//conv1: 5, 7, 5 pri: 5, 6, 0
    int tmp;
    int i, j, k, p, l, m;
    for (i = 0; i < output_width; i++) {
        for (j = 0; j < output_width; j++) {
            for (k = 0; k < output_ch; k++) {
                tmp = 0;
                for (p = 0; p < input_ch; p++) {
                    for (l = 0; l < kernel_width; l++) {
                        for (m = 0; m < kernel_width; m++) {
                            tmp += (input[input_width * input_width * p + input_width * (i * stride + l) + (j * stride + m)]*
                                        kernel[k * kernel_width * kernel_width * input_ch + p * kernel_width * kernel_width + l * kernel_width + m]);
                            /*tmp += mul_16(input[input_width * input_width * p + input_width * (i * stride + l) + (j * stride + m)],
                                        kernel[k * kernel_width * kernel_width * input_ch + p * kernel_width * kernel_width + l * kernel_width + m]);*/
                        }
                    }
                }
                if(stride==1)   output[k * output_width * output_width + i * output_width + j] = CLIP_8(tmp>>7);
                else            output[k * output_width * output_width + i * output_width + j] = CLIP_8(tmp>>11);
            }
        }
    }
}

void ReLU(char* input) {
    for (int i=0; i < conv1_out_ch * conv1_out_width * conv1_out_width; i++) {
        if (input[i] < 0) input[i] = 0;
    }
}
void squash(char* input, char* output, int dim_caps, int num_caps, int in_fraction, int out_fraction) {
    int tmp;
    for (int i=0; i < num_caps; i++) {
        tmp = 0;
        for (int j=0; j < dim_caps; j++) {
            tmp += abs(input[i * dim_caps + j]);
        }
        for (int k=0; k < dim_caps; k++) {
            output[i * dim_caps + k] = CLIP_8((input[i * dim_caps + k]<<(out_fraction)) / (tmp + (1<<in_fraction)));
        }
    }
}
void squash_(short* input, char* output, int dim_caps, int num_caps, int in_fraction, int out_fraction) {
    int tmp;
    for (int i=0; i < num_caps; i++) {
        tmp = 0;
        for (int j=0; j < dim_caps; j++) {
            tmp += abs(input[i * dim_caps + j]);
        }
        for (int k=0; k < dim_caps; k++) {
            output[i * dim_caps + k] = CLIP_8((input[i * dim_caps + k]<<(out_fraction)) / (tmp + (1<<in_fraction)));
        }
    }
}
void prediction_vectors(char* input, char* weight_matrix, short* output) {//7, 6, 16/13
    int tmp;
    for (int l = 0; l < num_primary_caps; l++) {
        for (int k = 0; k < num_class; k++) {
            for (int j = 0; j < dim_predic_vector; j++) {
                tmp = 0;
                for (int i = 0; i < dim_primary_caps; i++) {
                    tmp += (weight_matrix[l * dim_predic_vector * dim_primary_caps * num_class + dim_predic_vector * dim_primary_caps * k + dim_primary_caps * j + i] *
                            input[i + l * dim_primary_caps]);
                    /*output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] =
                        add_16(output[j + dim_predic_vector * k + dim_predic_vector * num_class * l],
                            CLIP_16(weight_matrix[l * dim_predic_vector * dim_primary_caps * num_class + dim_predic_vector * dim_primary_caps * k + dim_primary_caps * j + i] *
                                input[i + l * dim_primary_caps]));*/
                }
                output[j + dim_predic_vector * k + dim_predic_vector * num_class * l] = CLIP_16(tmp);
            }
        }
    }
}


void dynamic_routing(short* uhat, char* bias, char* v_j) {// 16/13, 10, 7

    short s_j[num_class * dim_predic_vector];
    //char u_vj1[num_primary_caps * num_class];
    int tmp = 0;
    int i, j, k, l;

    for (i = 0;i < num_iterations;i++) {
        //softmax(b_ij, c_ij);
        for (j = 0;j < num_class;j++) {
            for (k = 0;k < dim_predic_vector;k++) {
                tmp = 0;
                for (l = 0;l < num_primary_caps;l++) {
                    //tmp += ((c_ij[num_class * l + j] * uhat[num_class * dim_predic_vector * l + dim_predic_vector * j + k]));
                    tmp += uhat[num_class * dim_predic_vector * l + dim_predic_vector * j + k];
                }
                s_j[j * dim_predic_vector + k] = CLIP_16((tmp + bias[j * dim_predic_vector + k]<<3));// 16/13
                //v_j[j * dim_predic_vector + k] = CLIP_16(s_j[j * dim_predic_vector + k]);
            }
        }
        squash_(s_j, v_j, dim_predic_vector, num_class, 13, 7);
        /*
        for (int j = 0; j < dim_predic_vector; j++) {
            L1[j] = 0; L2[j] = 0; L8[j] = 0;
            for (int k = 0; k < num_class; k++) {
                //printf("s_j : %f\n", fx2fp(s_j[k * dim_predic_vector + j]));
                L1[j] = add_16(L1[j], mul_16(a_, qabs(s_j[k * dim_predic_vector + j])));
                if (qabs(s_j[k * dim_predic_vector + j]) > L8[j]) L8[j] = qabs(s_j[k * dim_predic_vector + j]);
            }
            tmp = mul_16(b_, L8[j]);
            L2[j] = add_16(L1[j], tmp);
            //printf(" L2 %d ",L2[j]);
            for (int k = 0; k < num_class; k++) {
                v_j[j + k * dim_predic_vector] = CLIP_16((s_j[j + k * dim_predic_vector] << fraction) / L2[j]);
            }
        }
        //squash(s_j, v_j, dim_predic_vector, num_class);
        for (j = 0; j < num_primary_caps; j++) {
            for (k = 0; k < num_class; k++) {
                tmp = 0;
                for (l = 0; l < dim_predic_vector; l++) {
                    tmp = add_16(tmp, mul_16(v_j[k * dim_predic_vector + l], uhat[j * num_class * dim_predic_vector + k * dim_predic_vector + l]));
                }
                b_ij[num_class * j + k] += tmp;
            }
        }*/
    }
}

int main() {
    FILE* fp, * num_wrong;
    int predict = 0;
    int max = 0;
    float accuracy;
    int predict_sum;
    int correct = 0;
    int tmp_predict = 0;
    int wrong = 0;
    char dataset[input_data_ch * input_data_width * input_data_width];
    char LABEL[data_size];

    fp = fopen("char_label_smallNORB.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(LABEL, sizeof(char), data_size, fp);
    printf("label : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("char_conv1_kernel_smallNORB.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv1_kernel, sizeof(char), conv1_out_ch * conv1_in_ch * kernel_width * kernel_width, fp);
    printf("conv1_kernel : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("char_conv2_kernel_smallNORB.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(conv2_kernel, sizeof(char), conv2_out_ch * conv2_in_ch * kernel_width * kernel_width, fp);
    printf("conv2_kernel : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("char_digit_caps_W_smallNORB.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(digits_W, sizeof(char), num_primary_caps * num_class * dim_predic_vector * dim_primary_caps, fp);
    printf("digits_W : OK\n");
    printf("\n");
    ///////////////////////////////////////////
    fp = fopen("char_digit_caps_bias_smallNORB.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }
    fread(digits_bias, sizeof(char), num_class * dim_predic_vector, fp);
    printf("digits_bias : OK\n");
    printf("\n");
    ///////////////////////////////////////////

    fp = fopen("char_dataset_smallNORB.bin", "rb"); if (fp == NULL){
        printf("Cannot open file.\n");
        exit(1);
    }

    for (int k = 0; k < data_size; k++) {
        fread(dataset, sizeof(char), input_data_ch * input_data_width * input_data_width, fp);
        printf("\n");
        for (int i = 0; i < 32*32; i++) {
            //printf("%x, ", conv2_kernel[i]);
        }
        
        convolution(dataset, conv1_kernel, conv1_output, conv1_in_width, conv1_in_ch, conv1_out_width, conv1_out_ch, conv1_stride);
        for (int i = 0; i < 147456; i++) {
            //printf("%x ", conv1_output[i]);
        }
        ReLU(conv1_output);
        //printf("\n");
        //printf("conv1 layer completed.\n");
        ///////////////////////////////////////////

        convolution(conv1_output, conv2_kernel, conv2_output, 24, 256, 8, 256, 2);
        for (int i = 0; i < 16384; i++) {
            //printf("%x ", conv2_output[i]);
        }
        //printf("\nconv2 layer completed.\n");
        ///////////////////////////////////////////
        char conv2_output_transpose[num_primary_caps * dim_primary_caps];
        for (int i = 0; i < dim_primary_caps; i++) {
            for (int j = 0; j < num_primary_caps; j++) {
                //conv2_output_transpose[j * dim_primary_caps + i] = conv2_output[i * num_primary_caps + j];
                conv2_output_transpose[i * num_primary_caps + j] = conv2_output[i * num_primary_caps + j];
            }
        }
        squash(conv2_output_transpose,conv2_output,dim_primary_caps,num_primary_caps, 0, 7);
        prediction_vectors(conv2_output, digits_W, u_hat);
        //printf("\n");
        //printf("\prediction_vectors completed.\n");
        ///////////////////////////////////////////

        dynamic_routing(u_hat, digits_bias, result_v);
        //printf("\dynamic routing completed.\n");
        ///////////////////////////////////////////



        max = 0;
        for (int i = 0; i < num_class; i++) {
            predict_sum = 0;
            for (int j = 0; j < dim_predic_vector; j++) {
                tmp_predict = (result_v[i * dim_predic_vector + j] * result_v[i * dim_predic_vector + j]);
                predict_sum += tmp_predict;
            }
            if (predict_sum > max) {
                max = predict_sum;
                predict = i;
            }
        }

        printf("(%d) pred : %d / ", k + 1, predict);
        if (predict == *(LABEL + k))  correct++;
        else {
            wrong++;
            num_wrong = fopen("error.txt", "wb");
            fprintf(num_wrong, "error : %d, wrong : %d\n", k, wrong);
            fclose(num_wrong);
        }
        printf("target: %d / ", *(LABEL + k));
        accuracy = (float)correct / (k + 1);
        printf("accuracy : %.2f\n\n", accuracy * 100);

    }
    fclose(fp);
    return 0;
}
