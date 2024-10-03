class Speakers:
    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    m1 = "m1"
    m2 = "m2"
    m3 = "m3"
    m4 = "m4"
    m5 = "m5"
    m6 = "m6"
    m7 = "m7"
    m8 = "m8"
    m9 = "m9"
    m10 = "m10"

    @staticmethod
    def get_all():
        return [
            Speakers.f1,
            Speakers.f2,
            Speakers.f3,
            Speakers.f4,
            Speakers.f5,
            Speakers.f6,
            Speakers.f7,
            Speakers.f8,
            Speakers.f9,
            Speakers.f10,
            Speakers.m1,
            Speakers.m2,
            Speakers.m3,
            Speakers.m4,
            Speakers.m5,
            Speakers.m6,
            Speakers.m7,
            Speakers.m8,
            Speakers.m9,
            Speakers.m10,
        ]


class Scripts:
    script1 = "script1"
    script2 = "script2"
    script3 = "script3"
    script4 = "script4"
    script5 = "script5"

    @staticmethod
    def get_all():
        return [
            Scripts.script1,
            Scripts.script2,
            Scripts.script3,
            Scripts.script4,
            Scripts.script5,
        ]


class Devices:
    ipad = "ipad"
    ipadflat = "ipadflat"
    iphone = "iphone"

    @staticmethod
    def get_all():
        return [Devices.ipad, Devices.iphone]


class Environments:
    office1 = "office1"
    office2 = "office2"
    confroom1 = "confroom1"
    confroom2 = "confroom2"
    livingroom1 = "livingroom1"
    bedroom1 = "bedroom1"
    balcony1 = "balcony1"
    clean = "clean"
    cleanraw = "cleanraw"
    produced = "produced"

    @staticmethod
    def get_all_noisy():
        return [
            Environments.office1,
            Environments.office2,
            Environments.confroom1,
            Environments.confroom2,
            Environments.livingroom1,
            Environments.bedroom1,
            Environments.balcony1,
        ]

    @staticmethod
    def get_all_clean():
        return [
            Environments.clean,
            Environments.cleanraw,
            Environments.produced,
        ]

    @staticmethod
    def get_all():
        return [
            *Environments.get_all_noisy(),
            *Environments.get_all_clean(),
        ]
