cd Datasets/giantsteps-key-dataset/audio/
file_names = ls;
num_files = size(file_names);
num_files = num_files(1);
file_names = file_names(3:num_files, :);
num_files = size(file_names);
num_files = num_files(1);
keys = [];

SR = 22025;

for i = 1: num_files
    name = file_names(i, :);
    audio = miraudio(name, "Sampling", SR);
    key = mirgetdata(mirkey(audio));
    keys = [keys key];
end
cd ../../../

writematrix(file_names, "file_names_key.txt");
writematrix(keys, "keys_pred.txt");