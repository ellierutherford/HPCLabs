void NonBlockedMultiply(double *mat1, double *mat2, double *result);
void PrintMatrix(double *result);
void InitializeMatrix(int seed, double *matrix);
void BlockedMultiply(double *mat1, double *mat2, double *result, int b);
void NonBlockedKij(double *mat1, double *mat2, double *result);
void MultiplyBlas(double *mat1, double *mat2, double *result);
void BlockedKijBlas(double *mat1, double *mat2, double *result, int b);
void BlockedKij(double *mat1, double *mat2, double *result, int b);
