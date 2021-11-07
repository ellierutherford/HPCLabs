#include <stdio.h>
#include <stdlib.h>
#include "MatrixMultiplication.h"
#include <sys/time.h>
#include<string.h>
#include <unistd.h>
#include <stdbool.h>

int n = 4;
bool TestBlockedMultiply(double *mat1, double *mat2, double *actualResult, double *expectedResult);
bool TestNonBlockedMultiply(double *mat1, double *mat2, double *actualResult,double *expectedResult);
bool TestBlockedBlas(double *mat1,double *mat2,double *actualResult,double *expectedResult);
bool TestBlas(double *mat1, double *mat2, double *actualResult, double *expectedResult);
bool TestKij(double *mat1,double *mat2,double *actualResult, double *expectedResult);
bool VerifyArraysEqual(double *actual, double *expected);
void PerformTests(double *mat1, double *mat2, double *actualResult, double *expectedResult);

int main(int argc,char* argv[]){

    double mat1[] = {136,158,112,122,100,123,96,32,160,102,175,27,163,93,104,164};
    double mat2[] = {6,1,8,8,6,9,9,4,9,1,2,6,7,2,5,7};
    double expectedResult[] = {3626, 1914, 3344, 3246, 2426, 1367, 2259, 2092, 3336, 1307, 2683, 2927, 3620, 1432, 3169, 3448};
    double *actualResult;
    actualResult = malloc(n*n*sizeof(double));

    printf("Perform first set of tests\n");
    PerformTests(mat1,mat2,actualResult,expectedResult);

    double mat3[] = {4,4,6,5,8,5,4,6,2,4,9,5,2,9,1,8};
    double mat4[] = {7,8,7,5,4,2,8,4,8,6,10,5,2,2,8,8};
    double expectedResult2[] = {102,86,160,106,120,110,184,128,112,88,176,111,74,56,160,115}; 

    printf("\nPerform second set of tests\n");
    PerformTests(mat3,mat4,actualResult,expectedResult2);

    double mat5[] = {22,17,20,16,21,9,19,19,15,8,17,7,23,17,3,12};
    double mat6[] = {20,6,17,10,20,23,13,8,8,16,4,3,8,22,5,12};
    double expectedResult3[] = {1068,1195,755,608,904,1055,645,567,652, 700,462, 349,920, 841,684, 519};

    printf("\nPerform third set of tests\n");
    PerformTests(mat5,mat6,actualResult,expectedResult3);

    free(actualResult);
}

void PerformTests(double *mat1, double *mat2, double *actualResult, double *expectedResult){
    char pass[] = "passed";
    char fail[] = "failed";

    // initialize the actual result matrix to all 0 elements before each test is performed
    InitializeMatrix(0,actualResult);
    bool testPass = TestBlockedMultiply(mat1,mat2,actualResult,expectedResult);
    printf("\nBlocked multiply %s\n", testPass ? pass : fail);
 
    InitializeMatrix(0,actualResult);
    testPass = TestNonBlockedMultiply(mat1,mat2,actualResult,expectedResult);
    printf("Non blocked multiply %s\n", testPass ? pass : fail);
    
    InitializeMatrix(0,actualResult);
    testPass = TestBlockedBlas(mat1,mat2,actualResult,expectedResult);
    printf("Blocked blas %s\n", testPass ? pass : fail);

    InitializeMatrix(0,actualResult);    
    testPass = TestBlas(mat1,mat2,actualResult,expectedResult);
    printf("Blas %s\n", testPass ? pass : fail);

    InitializeMatrix(0,actualResult);
    testPass = TestKij(mat1,mat2,actualResult,expectedResult); 
    printf("Kij %s\n", testPass ? pass : fail);
}

bool TestBlockedMultiply(double *mat1, double *mat2, double *actualResult, double *expectedResult){
    BlockedMultiply(mat1,mat2,actualResult,2);
    return VerifyArraysEqual(actualResult,expectedResult);
}

bool TestNonBlockedMultiply(double *mat1, double *mat2, double *actualResult, double *expectedResult){
    NonBlockedMultiply(mat1,mat2,actualResult);
    return VerifyArraysEqual(actualResult,expectedResult);
}

bool TestBlockedBlas(double *mat1,double *mat2,double *actualResult,double *expectedResult){
    BlockedKijBlas(mat1,mat2,actualResult,2);
    return VerifyArraysEqual(actualResult,expectedResult);
}

bool TestBlas(double *mat1, double *mat2, double *actualResult, double *expectedResult){
    MultiplyBlas(mat1,mat2,actualResult);
    return VerifyArraysEqual(actualResult,expectedResult);
}

bool TestKij(double *mat1,double *mat2,double *actualResult, double *expectedResult){
   NonBlockedKij(mat1,mat2,actualResult);
   return VerifyArraysEqual(actualResult,expectedResult); 
}


bool VerifyArraysEqual(double *actual, double *expected){
    for(int i=0;i<n*n;i++){
        if(actual[i]!=expected[i]){
	    return false;
	}
    }
    return true;
}
