#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define ETA 0.000001
#define C 100.0
#define EPSILON 0.000001

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


// Compute w * phi for every feature vector
double *compute_scores(double *w, double **features, int num_samples, int num_features) {
  double *scores = (double *)malloc(num_samples * sizeof(double));
  int sample_ind, other_sample_ind;
  
  for (sample_ind=0; sample_ind<num_samples; ++sample_ind) {
    scores[sample_ind] = dot_product(w, features[sample_ind], num_features);
  }
  
  return scores;
}


double compute_objective(double *w, double **features, int *grades, int num_samples, int num_features) {
  double slack_sum = 0;
  double *scores = (double *)malloc(num_samples * sizeof(double));
  int sample_ind, other_sample_ind;
  
  for (sample_ind=0; sample_ind<num_samples; ++sample_ind) {
    scores[sample_ind] = dot_product(w, features[sample_ind], num_features);
    for (other_sample_ind=0; other_sample_ind<sample_ind; ++other_sample_ind) {
      if (grades[sample_ind] < grades[other_sample_ind] && scores[sample_ind]+1 > scores[other_sample_ind]) {
        slack_sum += 1 + scores[sample_ind] - scores[other_sample_ind];
      } else if (grades[sample_ind] > grades[other_sample_ind] && scores[sample_ind] < 1+scores[other_sample_ind]) {
        slack_sum += 1 + scores[other_sample_ind] - scores[sample_ind];
      }
    }
  }
  free(scores);
  
  return dot_product(w, w, num_features) / 2 + slack_sum * C / (num_samples * num_samples);
}


// Do an iteration of stochastic gradient descent and return the post-update objective function
double stochastic_gradient_descent(double *w, double **features, int *grades, int num_samples, int num_features) {
  double *scores = (double *)malloc(num_samples * sizeof(double));
  int sample_ind, other_sample_ind;
  double step_size = C*ETA/pow(num_samples, 2);

  for (sample_ind=0; sample_ind<num_samples; ++sample_ind) {
    scores[sample_ind] = dot_product(w, features[sample_ind], num_features);
    for (other_sample_ind=0; other_sample_ind<sample_ind; ++other_sample_ind) {
      if (grades[sample_ind] < grades[other_sample_ind] && scores[sample_ind]+1 > scores[other_sample_ind]) {
        vec_assign(w, w, 1-ETA, num_features);
        vec_add(w, features[sample_ind], -step_size, num_features);
        vec_add(w, features[other_sample_ind], step_size, num_features);
      } else if (grades[sample_ind] > grades[other_sample_ind] && scores[sample_ind] < 1+scores[other_sample_ind]) {
        vec_assign(w, w, 1-ETA, num_features);
        vec_add(w, features[sample_ind], step_size, num_features);
        vec_add(w, features[other_sample_ind], -step_size, num_features);
      }
    }
  }
  free(scores);
  
  return compute_objective(w, features, grades, num_samples, num_features);
}

// Return a list of indices that sorts the features into increasing-score order
int *rank_features(double *w, double **features, int num_samples, int num_features) {
  double *scores = compute_scores(w, features, num_samples, num_features);
  int *sample_inds = (int *)malloc(num_samples * sizeof(int));
  
  int compare_samples(const void *ind_a, const void *ind_b) {
    double temp = scores[*(int *)ind_a] - scores[*(int *)ind_b];
    if (temp > 0)
      return 1;
    else if (temp < 0)
      return -1;
    else
      return 0;
  }
  qsort(sample_inds, num_samples, sizeof(int), compare_samples);
  
  free(scores);
  
  return sample_inds;
}


// rank_svm data_file model_file
int main(int argc, char *argv[]) {
  char *data_filename, *model_filename, *scores_filename;
  char line[1024];
  FILE *fp;
  double **features;
  int *grades;
  double *w;
  double *scores; // w * phi for every feature vector phi
  int num_features, num_samples;
  int sample_ind, other_sample_ind, feature_ind;
  int tmp_int1, tmp_int2, iter_cnt;
  double best_objective, cur_objective;
  
  if (argc == 3) {
    printf("Training on %s, writing model to %s\n", argv[1], argv[2]);
  } else if (argc == 4) {
    printf("Evaluating on %s, using model %s, writing scores to %s\n", argv[1], argv[2], argv[3]);
    scores_filename = argv[3];
  } else {
    printf("Usage: rank_svm data_file model_file OR rank_svm test_file model_file score_file\n");
    printf("You entered %d args\n", argc);
    return 0;
  }
  data_filename = argv[1];
  model_filename = argv[2];
  
  fp = fopen(data_filename, "r");
  fgets(line, 1024, fp);
  parse_header(line, &num_samples, &num_features);
  
  features = allocate(num_samples, num_features);
  w = (double *)malloc(num_features * sizeof(double));
  scores = (double *)malloc(num_samples * sizeof(double));
  grades = (int *)malloc(num_samples * sizeof(int));
  
  for (sample_ind=0; sample_ind < num_samples; ++sample_ind) {
    fgets(line, 1024, fp);
    parse_row(line, grades+sample_ind, features[sample_ind], num_features);
  }
  fclose(fp);
  

  if (argc == 3) {
    // Train
    for (feature_ind=0; feature_ind<num_features; ++feature_ind) {
      w[feature_ind] = 0;
    }
    
    iter_cnt = 0;
    best_objective = cur_objective = compute_objective(w, features, grades, num_samples, num_features);
    printf("Iteration %d: objective %f\n", iter_cnt, cur_objective);

    cur_objective = stochastic_gradient_descent(w, features, grades, num_samples, num_features);
    ++iter_cnt;
    printf("Iteration %d: objective %f\n", iter_cnt, cur_objective);
    while (cur_objective < best_objective - EPSILON) {
      best_objective = cur_objective;
      cur_objective = stochastic_gradient_descent(w, features, grades, num_samples, num_features);
      ++iter_cnt;
      printf("Iteration %d: objective %f\n", iter_cnt, cur_objective);
    }
    
    fp = fopen(model_filename, "w");
    fprintf(fp, "1\t%d\n", num_features);
    fprintf(fp, "0");
    for (feature_ind=0; feature_ind<num_features; ++feature_ind) {
      fprintf(fp, "\t%f", w[feature_ind]);
    }
    fprintf(fp, "\n");
    fclose(fp);
  } else if (argc == 4) {
    // Test
    fp = fopen(model_filename, "r");
    fgets(line, 1024, fp);
    parse_header(line, &tmp_int1, &tmp_int2);
    if (tmp_int2 != num_features) {
      printf("ERROR: model has %d features, data has %d\n", tmp_int2, num_features);
      return 1;
    }
  
    fgets(line, 1024, fp);
    parse_row(line, &tmp_int1, w, num_features);
      
    fclose(fp);
    
    scores = compute_scores(w, features, num_samples, num_features);
    fp = fopen(scores_filename, "w");
    for (sample_ind=0; sample_ind<num_samples; ++sample_ind) {
      fprintf(fp, "%f\n", scores[sample_ind]);
    }
    fclose(fp);
  }
  
  free(grades);
  free(w);
  deallocate(features, num_samples);
  
  return 0;
}


































