function mat = matNorm(mat)
%Normalizes a matrix so it's between 0 and 1
mat = mat-min(mat(:));
mat = mat./max(mat(:));
