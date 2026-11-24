set V;   # vertices in the original graph
param m; # number of cycles
set C := 0..m-1;
set CYCLE{C};   # vertices in each cycle
param w{c in C} default card(CYCLE[c]);
param min_tx default 0;  # minimum number of transplants required

var x{C} binary;

maximize z: sum {c in C} w[c] * x[c];

Packing {i in V}: sum {c in C: i in CYCLE[c]} x[c] <= 1;
MinTransplants: sum {c in C} card(CYCLE[c]) * x[c] >= min_tx;