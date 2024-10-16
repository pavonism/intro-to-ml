class Environments:
    bed = "bed"
    bird = "bird"
    cat = "cat"
    dog = "dog"
    down = "down"
    eight = "eight"
    five = "five"
    four = "four"
    go = "go"
    happy = "happy"
    house = "house"
    left = "left"
    marvin = "marvin"
    nine = "nine"
    no = "no"
    off = "off"
    on = "on"
    one = "one"
    right = "right"
    seven = "seven"
    sheila = "sheila"
    six = "six"
    stop = "stop"
    three = "three"
    tree = "tree"
    two = "two"
    up = "up"
    wow = "wow"
    yes = "yes"
    zero = "zero"
    background_noise = "_background_noise_"

    @staticmethod
    def get_all_noisy():
        return [Environments.background_noise]

    @staticmethod
    def get_all_clean():
        return [
            Environments.bed,
            Environments.bird,
            Environments.cat,
            Environments.dog,
            Environments.down,
            Environments.eight,
            Environments.five,
            Environments.four,
            Environments.go,
            Environments.happy,
            Environments.house,
            Environments.left,
            Environments.marvin,
            Environments.nine,
            Environments.no,
            Environments.off,
            Environments.on,
            Environments.one,
            Environments.right,
            Environments.seven,
            Environments.sheila,
            Environments.six,
            Environments.stop,
            Environments.three,
            Environments.tree,
            Environments.two,
            Environments.up,
            Environments.wow,
            Environments.yes,
            Environments.zero,
        ]

    @staticmethod
    def get_all():
        return [
            *Environments.get_all_noisy(),
            *Environments.get_all_clean(),
        ]
