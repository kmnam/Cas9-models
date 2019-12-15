#!/usr/bin/env wolframscript
(* ::Package:: *)

(*--
Symbolic calculations with the line graph.

Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/12/2019
--*)

computeCleavageStats[L_, N_, kdis_, kcat_] := Module[
    {prob, time, stats, bi, di, t, t1, t2, u1, u2},
    bi[i_] := If[0 < i < N, L[[i+1,1]], kcat];
    di[i_] := If[0 < i <= N, L[[i,2]], kdis];
    prob = 0;
    For[i = 0, i <= N, i++; t = 1;
        For[j = 0, j <= i, j++; t = t * di[j]/bi[j]];
        prob = prob + t;
    ];
    prob = 1/prob;
    time = 0;
    For[i = 0, i <= N, i++; t1 = 1; t2 = 1;
        For[j = i + 1, j <= N, j++; u1 = 1;
            For[k = i + 1, k <= j, k++; u1 = u1 * di[k]/bi[k]];
            t1 = t1 + u1;
        ];
        For[j = 0, j < i, j++; u2 = 1;
            For[k = 0, k <= j, k++; u2 = u2 * di[k]/bi[k]];
            t2 = t2 + u2;
        ];
        time = time + (t1 * t2)/bi[i];
    ];
    time = time * prob;
    stats = {prob, time};
    Return[stats];
];
labelsMatch = ConstantArray[{x^p, x^-q}, 20];
labelsMismatch1 = Join[ConstantArray[{x^p, x^-q}, 19], {{x^q, x^-p}}];
statsMatch = computeCleavageStats[labelsMatch, 20, 1, 1]
statsMismatch1 = computeCleavageStats[labelsMismatch1, 20, 1, 1]
leadingTerm[f_, x_] := Coefficient[f[x], x^(Exponent[f[x], x])] * x^(Exponent[f[x], x]);




