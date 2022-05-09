function [st,sv] = build_space_domain_fun(v_ref)
%Summary of this function goes here
%   convert velocity profile in time domain into space domain
%   call: [st,sv] = build_space_domain_fun(); in main file
%   input: v vs t profile, output: v vs s and t vs s profiles;
%   Note: the velocity inported should not contain zero velocity druing the period
%   Note: the velocity is round to integers.
%   creater: Sheng Yu
%   date: 18/09/2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load reference data in time domain               
% open('WLTP1.fig')
% b = get(gca,'Children');
% y= get(b, 'YData');           % the above code is to input reference vehicle speed.
% v_ref=y(1,107:355); % non stop piece(107:355) WLTP1 Low 1 1:589
% v_ref=y(1,355:414);
% v_ref=[1,2,2.22,3.18,4,2,1,1,1];
v_ref=round(v_ref);
v_ref=v_ref(v_ref~=0);   % remove beginning zeros
N=size(v_ref,2);         % Number of data taken from the leading vehicle
dis_ref=[];
dis_ref(1)=v_ref(1);  
for i=1:N-1
    dis_ref(i+1)=dis_ref(i)+v_ref(i+1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% time vs distance
st=zeros(1,round(dis_ref(end)));
t_ref=[1:N];
for i=1:size(dis_ref,2)
    if dis_ref(i)==0
        st=st+ones(1,size(st,2));
    else
        st(fix(dis_ref(i)))=i;
        if i>1 && dis_ref(i)~=dis_ref(i-1)+1
            s_gap=dis_ref(i)-dis_ref(i-1);
            interval=1/s_gap;
            for k=dis_ref(i-1)+1:dis_ref(i)-1
                st(fix(k))=st(fix(k-1))+interval;
            end
        end
    end
end
% figure()
% subplot(2,1,1)
% plot(dis_ref,t_ref,'.')
% hold on
% plot(st(1,:),'--')
% grid on
% legend('time domain','space domain')
% title('time vs distance by time domain and space domain data')
% xlabel('distance [m]')
% ylabel('time [s]')
%%%%%space domain%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=[];
sv=zeros(1,round(dis_ref(end)));
for i=1:size(dis_ref,2)
    sv(fix(dis_ref(i)))=v_ref(i);
end
for j=size(sv,2):-1:1
    if sv(j)==0
        sv(j)=sv(j+1);
    end
end
% subplot(2,1,2)
% plot(dis_ref,v_ref,'.')
% hold on
% plot(sv(1,:))
% grid on
% legend('time domain','space domain')
% title('velocity vs distance by time domain and space domain data')
% xlabel('distance [m]')
% ylabel('velocity [m/s]')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

