cd Datasets/ballroom/
tempos = [];
all_file_names = [];
dirs = ["Chacha", "Jive", "Quickstep", "Rumba", "Samba", "Tango", "Viennesewaltz", "Slowwaltz"];

SR = 11025;

for i = 1: length(dirs)
    cd (dirs(i))
    file_names = ls;
    num_files = size(file_names);
    num_files = num_files(1);
    file_names = file_names(3:num_files, :);
    num_files = size(file_names);
    num_files = num_files(1);

    for j = 1: num_files
        name = file_names(j, :);
        audio = miraudio(name, "Sampling", SR);
        key = mirgetdata(mirtempo(audio));
        keys = [keys key];
        all_file_names = [all_file_names name];
    end
    cd ..
end
cd ../../

writematrix(all_file_names, "file_names_tempo.txt");
writematrix(keys, "tempos.txt");