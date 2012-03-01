#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define ETA 0.03
#define LAMBDA 0.2

// a * b
double dot_product(double *a, double *b, int n) {
  double dp = 0; int i;
  for (i=0; i<n; ++i) dp+=a[i]*b[i];
  return dp;
}

// a += lambda * b
void vec_add(double *a, double *b, double lambda, int n) {
  int i;
  for (i=0; i<n; ++i) a[i]+=lambda*b[i];
}

// a = lambda * b
void vec_assign(double *a, double *b, double lambda, int n) {
  int i;
  for (i=0; i<n; ++i) a[i]=lambda*b[i];
}


void switch_ptrs(double ***a, double ***b) {
  double **tmp = *a;
  *a = *b;
  *b = tmp;
}

void print_vec(double *a, int n) {
  int i;
  for (i=0; i<n; ++i) printf("%f  ", a[i]);
  printf("\n");
}


// allocate double ** matrix of x rows and y columns and set to zero
double **allocate(int x, int y) {
  int i, k;
  double **tmp = (double **)malloc(x * sizeof(double *));
  for (i=0; i<x; ++i) {
    tmp[i] = (double *)malloc(y * sizeof(double));
    for (k=0; k<y; ++k) tmp[i][k] = 0;
  }
  return tmp;
}
void deallocate(double **p, int x) {
  int i;
  for (i=0; i<x; ++i) free(p[i]);
  free(p);
}



// Parse a data file header: num_samples<tab>num_features\n
void parse_header(const char *row, int *num_samples, int *num_features) {
  char const *cur_char = row;
  char tmp_str[32];
  int num_chars;
  int feat_ind;
  
  num_chars = 0;
  while (*cur_char != '\t') { ++cur_char; ++num_chars; }
  memcpy(tmp_str, row, num_chars);
  tmp_str[num_chars] = 0;
  *num_samples = atoi(tmp_str);
  
  num_chars = 0; ++cur_char;
  while (*cur_char != '\n' && *cur_char != 0) { ++cur_char; ++num_chars; }
  memcpy(tmp_str, cur_char-num_chars, num_chars);
  tmp_str[num_chars] = 0;
  *num_features = atoi(tmp_str);
}

// Parse a row: grade<tab>feature 1<tab>feature 2... feature n\n
void parse_row(const char *row, int *grade, double *features, int n) {
  char const *cur_char = row;
  char tmp_str[32];
  int num_chars;
  int feat_ind;
  
  num_chars = 0;
  while (*cur_char != '\t') { ++cur_char; ++num_chars; }
  memcpy(tmp_str, row, num_chars);
  tmp_str[num_chars] = 0;
  *grade = atoi(tmp_str);
  
  for (feat_ind=0; feat_ind<n; ++feat_ind) {
    num_chars = 0; ++cur_char;
    while (*cur_char != '\t' && *cur_char != '\n' && *cur_char != 0) { ++cur_char; ++num_chars; }
    memcpy(tmp_str, cur_char-num_chars, num_chars);
    tmp_str[num_chars] = 0;
    features[feat_ind] = atof(tmp_str);    
  }
}


// rank_svm data_file model_file
int main(int argc, char *argv[]) {
  char *train_filename;
  char line[1024];
  FILE *fp;
  double **features;
  int *grades;
  double *w;
  int num_features, num_samples, sample_ind;
  
  if (argc != 3) {
    printf("Usage: rank_svm data_file model_file\n");
    printf("You entered %d args\n", argc);
    return 0;
  }
  train_filename = argv[1];
  
  fp = fopen(train_filename, "r");
  fgets(line, 1024, fp);
  parse_header(line, &num_samples, &num_features);
  
  features = allocate(num_samples, num_features);
  grades = (int *)malloc(num_samples * sizeof(int));
  
  printf("Here are the %d grades and features:\n", num_samples);
  for (sample_ind=0; sample_ind < num_samples; ++sample_ind) {
    fgets(line, 1024, fp);
    parse_row(line, grades+sample_ind, features[sample_ind], num_features);
    printf("%d ", grades[sample_ind]);
    print_vec(features[sample_ind], num_features);
  }
  fclose(fp);
  
  free(grades);
  deallocate(features, num_samples);
  
  return 0;
}


































