#!/usr/bin/env wolframscript
(* ::Package:: *)

(*--
Symbolic calculations with the grid graph.

Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/12/2019
--*)

recurA[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{l*c+k*d+c*d, a*k*d, b*l*c}, {d, a*d, 0}, {c, 0, b*c}}.v
];
recurB[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{a*(l+d), b*l, a*b*l}, {a*k, b(k+c), a*b*k}, {a, b, a*b}}.v
];
recurC[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{l*c+k*d+c*d, 0, b*l*c}, {d, 0, 0}, {c, 0, b*c}}.v
];
recurD[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{0, l*c+k*d+c*d, a*k*d}, {0, d, a*d}, {0, c, 0}}.v
];
recurE[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{d+l, a(d+l), b*l, a*b*l}, {c+k, a*k, b(c+k), a*b*k}, {1, a, b, a*b}}.v
];
recurF[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{d+l, b*l}, {c+k, b(c+k)}, {1, b}}.v
];
recurG[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{d+l, a(d+l)}, {c+k, a*k}, {1, a}}.v
];
recurH[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{d+l, a(d+l), b*l, a*b*l}, {c+k, a*k, b(c+k), a*b*k}, {1, a, b, a*b}}.v
];
recurI[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{0, b, a*b, 0}, {a, 0, 0, a*b}}.v
];
recurJ[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{d+l, 0, 0, b*l}, {0, d+l, a(d+l), 0}, {c+k, 0, 0, b(c+k)}, {0, c+k, a*k, 0}}.v
];
recurK[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{1, 0, 0, b}, {0, 1, a, 0}}.v
];
recurL[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{a(d+l), b*l, a*b*l, a*b*l}, {a*k, b(c+k), a*b*k, a*b*k}, {a, 0, a*b, 0}, {0, b, 0, a*b}}.v
];
recurM[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{k, a*k, b*k, a*b*k, b*c, 0}, {l, a*l, b*l, a*b*l, 0, a*d}}.v
];
recurN[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{a(d+l), 0, a*b*l, b*l, 0}, {0, b*l, a*b*l, 0, a(d+l)},
    {a*k, 0, a*b*k, b(c+k), 0}, {0, b(c+k), a*b*k, 0, a*k}}.v
];
recurO[L_, v_, j_] := Module[{a, b, c, d, k, l},
    a = L[[j,1]];
    c = L[[j,2]];
    b = L[[j,3]];
    d = L[[j,4]];
    k = L[[j,5]];
    l = L[[j,6]];
    {{0, 0, 0, b, 0}, {a, 0, a*b, 0, 0}, {0, b, a*b, 0, 0}, {0, 0, 0, 0, a}}.v
];
weightTrees[L_, k0_, l0_, x_, i_, N_] := Module[{v},
    v = {l0, k0, 1};
    For[j = 1, j <= i, j++, v = recurB[L, v, j]];
    If[i == N, If[x == 0, Return[v[[1]]], Return[v[[2]]]]];
    If[x == 0, v = recurC[L, v, i+1], v = recurD[L, v, i+1]];
    For[j = i+2, j <= N, j++, v = recurA[L, v, j]];
    Return[v[[1]]];
];
weightEndToEndForests[L_, k0_, l0_, N_] := Module[{a, b, c, d, k, l, u, v, w},
    a = L[[1,1]];
    c = L[[1,2]];
    b = L[[1,3]];
    d = L[[1,4]];
    k = L[[1,5]];
    l = L[[1,6]];
    u = {l0*c*d+l0*l*c+b*l*c+k*d*l0, d*l0, b*c+c*l0};
    w = {u[[1]], b*l+d*l0+l0*l, b*k+c*l0+b*c+l0*k, b+l0};
    For[j = 2, j <= N, j++, v = recurE[L, w, j];
        u = recurA[L, u, j];
        w = {u[[1]], v[[1]], v[[2]], v[[3]]}
    ];
    Return[v[[2]]];
];
weightDirectedForests[L_, k0_, l0_, x_, i_, N_] := Module[{a, b, c, d, e, u, v, w},
    a = {l0, k0, 1};
    b = {0, 1};
    For[j = 1, j <= i, j++, c = {a[[1]], a[[2]], b[[1]], b[[2]]};
        b = recurI[L, c, j];
        a = recurB[L, a, j]
    ];
    If[x == 0 || x == N, Return[b[[2]]]];
    c = {a[[1]], a[[2]], b[[1]], b[[2]]};
    d = recurJ[L, c, i+1];
    If[i == N-1, If[x == 0, Return[d[[3]]], Return[d[[4]]]]];
    e = recurK[L,c,i+1];
    If[x == 0, u = recurC[L, a, i+1], u = recurD[L, a, i+1]];
    If[x == 0, w = {u[[1]], d[[1]], d[[3]], e[[1]]}, w = {u[[1]], d[[2]], d[[4]], e[[2]]}];
    For[j = i+2, j <= N, j++, v = recurH[L, w, j];
        u = recurA[L, u, j];
        w = {u[[1]], v[[1]], v[[2]], v[[3]]}
    ];
    Return[v[[1]]];
];
weightDirectedForests2[L_, k0_, l0_, x_, i_, N_] := Module[{a, b, c, d, e, f, g, h, v},
    a = {l0, 0, 1};
    b = {l0, 0, 1, 0};
    c = {l0, 0, 1, 0, 0, 0};
    d = {0, 1, 0};
    e = {0, 0};
    For[j = 1, j <= i, j++, e = recurM[L, c, j];
        a = recurA[L, a, j];
        d = recurE[L, b, j];
        b = {a[[1]], d[[1]], d[[2]], d[[3]]};
        c = {a[[1]], d[[1]], d[[2]], d[[3]], e[[1]], e[[2]]}
    ];
    If[x == 0 && i == N, Return[e[[1]]]];
    f = {d[[1]], d[[2]], d[[3]], e[[1]], e[[2]]};
    g = recurN[L, f, i+1];
    h = recurO[L, f, i+1];
    If[i == N-1, If[x == 0, Return[g[[3]]], Return[g[[4]]]]];
    If[x == 0, v={g[[1]], g[[3]], h[[1]], h[[2]]}, v = {g[[2]], g[[4]], h[[3]], h[[4]]}];
    For[j = i+2, j <= N, j++, v = recurL[L, v, j]];
    Return[v[[2]]];
];
computeCleavageStats[L_, k0_, l0_, N_, kdis_, kcat_] :=
    Module[{wt, wf, wr, wA0BN, denom, prob, time, term, stats},
    wt = ConstantArray[0,{N+1,2}];
    For[i = 0, i <= N, i++,
        wt[[i+1,1]] = weightTrees[L,k0,l0,0,i,N];
        wt[[i+1,2]] = weightTrees[L,k0,l0,1,i,N]
    ];
    wA0BN = weightEndToEndForests[L,k0,l0,N];
    wf = ConstantArray[0,{N+1,2}];
    For[i = 1, i <= N, i++, wf[[i+1,1]] = weightDirectedForests[L,k0,l0,0,i,N]];
    For[i = 0, i < N, i++, wf[[i+1,2]] = weightDirectedForests[L,k0,l0,1,i,N]];
    wr = ConstantArray[0,{N+1,2}];
    For[i = 1, i <= N, i++, wr[[i+1,1]] = weightDirectedForests2[L,k0,l0,0,i,N]];
    For[i = 0, i < N, i++, wr[[i+1,2]] = weightDirectedForests2[L,k0,l0,1,i,N]];
    denom = kdis*wt[[1,1]] + kcat*wt[[N+1,2]] + kdis*kcat*wA0BN;
    prob = (kcat*wt[[N+1,2]])/denom;
    time = 0;
    For[i = 1, i <= N, i++,
        term = (kcat*wf[[i+1,1]] + wt[[i+1,1]])/denom;
        term = term (1 + (kdis*wr[[i+1,1]])/wt[[N+1,2]]);
        time = time + term;
    ];
    For[i = 0, i < N, i++,
        term = (kcat*wf[[i+1,2]] + wt[[i+1,2]])/denom;
        term = term (1 + (kdis*wr[[i+1,1]])/wt[[N+1,2]]);
        time = time + term;
    ];
    time = time + (wt[[1,1]] + wt[[N+1,2]] + wA0BN(kdis+kcat))/denom;
    stats = {prob, time};
    Return[stats];
];
labelsMatch = ConstantArray[{100000,1/100000,100000,1/100000,1/100000,100000},20];
labelsMismatch = ConstantArray[{1/100000,100000,1/100000,100000,1/100000,100000},20];
statsMatch = computeCleavageStats[labelsMatch,1/100000,100000,20,1,1];
statsMismatch = computeCleavageStats[labelsMismatch,1/100000,100000,20,1,1];
N[Log10[statsMatch[[1]]] - Log10[statsMismatch[[1]]]]
N[Log10[statsMismatch[[2]]] - Log10[statsMatch[[2]]]]
