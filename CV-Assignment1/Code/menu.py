
class KmeansMenu:

    def __init__(self, rvalue=0, again=0, kval=1):
        self.rvalue = rvalue
        self.kval = kval
        self.again = again

    def prompt(self):
        condition = False

        while condition == False:
            val = int(input(
                "Enter an r value (1-100) where r is the number of times the algorithm executes: "))

            if val < 1 or val > 100:
                print("Invalid r value. Please try again")

            else:
                kval = int(input(
                    "Enter the number of K clusters (1-25): "))

                if kval < 1 or kval > 25:
                    print("Invalid k value. Please try again")

                else:
                    condition = True

        self.rvalue = val
        self.kval = kval
