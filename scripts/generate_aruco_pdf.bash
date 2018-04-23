for i in $(seq 0 9); do 
../bin/create_marker -d=6 -id=$i --ms=493 aruco00$i.png
done;
for i in $(seq 10 99); do 
../bin/create_marker -d=6 -id=$i --ms=493 aruco0$i.png
done;
for i in $(seq 100 249); do 
../bin/create_marker -d=6 -id=$i --ms=493 aruco$i.png
done;

# 18 mm margin, 174 mm marker
convert -page a4 -border 51 -bordercolor white -gravity north *.png out.pdf

