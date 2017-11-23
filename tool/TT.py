import csv

with open ("../NASDAQ/price_NASDAQ100_20000101_20150101.csv", 'r') as f:
    with open('../NASDAQ/price.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['NDX'])

        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            price = row[1]
            print row[1]
            writer.writerow([row[1]])

        


