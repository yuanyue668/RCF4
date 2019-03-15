function results=run_RHCF(seq, res_path, bSaveImage)

close all;

x=seq.init_rect(1);
y=seq.init_rect(2);
w=seq.init_rect(3);
h=seq.init_rect(4);

tic
command = sprintf('python rcf4_tracker.py %s %s %s %s %s %s %s %s',seq.name,seq.path,num2str(seq.startFrame),num2str(seq.endFrame),num2str(x),num2str(y),num2str(w),num2str(h));
system(command)
dos(command);
duration=toc;

results.res = dlmread([seq.name '_ST.txt']);
results.type='rect';
results.fps=seq.len/duration;
