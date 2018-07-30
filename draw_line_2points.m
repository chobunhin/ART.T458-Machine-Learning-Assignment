function  draw_line_2points( A, B, s_color )
  xlim = get(gca,'XLim');
  m = (B(2)-B(1))/(A(2)-A(1));
  n = B(2)*m - A(2);
  y1 = m*xlim(1) + n;
  y2 = m*xlim(2) + n;
  hold on
  line([xlim(1) xlim(2)],[y1 y2], 'Color', s_color);

end

