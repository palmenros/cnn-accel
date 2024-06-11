rm -f outputDogs.txt
rm -f outputCats.txt

for i in images/dog.*; do echo $i; ./cnnSolver $i | grep OUTPUT >> outputDogs.txt ; done
for i in images/cat.*; do echo $i; ./cnnSolver $i | grep OUTPUT >> outputCats.txt ; done

echo -n "Positive dogs: "; grep "DOG" outputDogs.txt | wc -l
echo -n "Negative dogs: "; grep "CAT" outputDogs.txt | wc -l

echo ""

echo -n "Positive cats: "; grep "CAT" outputCats.txt | wc -l
echo -n "Negative cats: "; grep "DOG" outputCats.txt | wc -l

