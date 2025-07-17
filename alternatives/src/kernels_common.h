#if !defined(SINGLE_PRECISION) && !defined(DOUBLE_PRECISION)
    // Default to single precision
    #define SINGLE_PRECISION
#endif

#if defined(DOUBLE_PRECISION)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#if defined(SINGLE_PRECISION)
    typedef float real;
    #define ZERO 0.0f
    #define ONE 1.0f
#elif defined(DOUBLE_PRECISION)
    typedef double real;
    #define ZERO 0.0
    #define ONE 1.0
#endif
