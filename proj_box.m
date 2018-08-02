function [ x ] = proj_box( x, a, b )
%PROJ_BOX project x onto [a,b]
if a>=b
  return;
end
x = max(x,a);
x = min(x,b);


end

