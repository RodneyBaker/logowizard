"""Curated font families from the Google Fonts repository.

Each family is tagged with a shape class (what the letters look like) and
moods (what kind of title it suits). The fetcher looks the family up in the
ofl/, apache/ and ufl/ folders, so only openly licensed fonts can come in.

Tags:
  class: serif, sans, slab, script, display, blackletter, hand, mono
  mood:  comic, novel, horror, western, scifi, fantasy, retro, elegant, pulp, playful
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyEntry:
    dirname: str
    font_class: str
    moods: tuple[str, ...]


def _entries(block: str) -> list[FamilyEntry]:
    out = []
    for line in block.strip().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        dirname, font_class, *moods = line.split()
        out.append(FamilyEntry(dirname, font_class, tuple(moods)))
    return out


FAMILIES = _entries(
    """
    # Heavy comic display
    bangers          display comic pulp
    luckiestguy      display comic playful
    bowlbyone        display comic retro
    bowlbyonesc      display comic retro
    titanone         display comic playful
    lilitaone        display comic playful
    chango           display comic retro
    bungee           display comic scifi
    bungeeshade      display comic scifi
    bungeeinline     display comic retro
    rammettoone      display comic playful
    sigmarone        display comic retro
    passionone       display comic pulp
    blackopsone      display comic pulp
    russoone         display comic scifi
    righteous        display comic retro
    monoton          display comic retro
    fasterone        display comic retro
    racingsansone    display comic pulp
    kavoon           display comic playful
    lemon            display comic playful
    shojumaru        display comic fantasy
    chewy            display comic playful
    bubblegumsans    display comic playful
    boogaloo         display comic playful
    carterone        display comic playful
    knewave          display comic pulp
    ultra            slab    comic pulp western
    holtwoodonesc    slab    comic western
    alfaslabone      slab    comic pulp
    anton            sans    comic pulp
    bebasneue        sans    comic pulp scifi
    oswald           sans    comic pulp
    fjallaone        sans    comic pulp
    archivoblack     sans    comic pulp
    staatliches      display comic pulp
    squadaone        display comic scifi
    permanentmarker  hand    comic playful
    rubikmonoone     display comic scifi
    delagothicone    display comic scifi
    dynapuff         display comic playful
    gluten           display comic playful
    rowdies          display comic playful
    modak            display comic playful
    pollerone        display comic retro
    coiny            display comic playful
    galindo          display comic playful
    sniglet          display comic playful
    chicle           display comic retro
    bigshotone       display comic retro
    ranchers         display comic western playful
    lalezar          display comic pulp
    fascinate        display comic retro
    fascinateinline  display comic retro
    bevan            slab    comic retro
    poetsenone       display comic playful
    bagelfatone      display comic playful
    rubikmoonrocks   display comic scifi
    rubikglitch      display comic scifi
    rubikbeastly     display comic horror
    rubikwetpaint    display comic horror
    bungeespice      display comic retro
    comicneue        hand    comic playful
    bangers          display comic pulp
    slackey          display comic playful
    fontdinerswanky  display comic retro
    cherrycreamsoda  display comic retro
    irishgrover      display comic playful
    luckiestguy      display comic playful
    frijole          display comic western horror
    sancreek         display comic western
    smokum           display comic western
    rye              display comic western
    ewert            display comic western
    diplomata        display comic western retro
    wellfleet        slab    comic western
    jollylodger      display comic fantasy
    # Horror
    creepster        display comic horror
    butcherman       display comic horror
    nosifer          display comic horror
    eater            display comic horror
    metalmania       display comic horror
    newrocker        display comic horror fantasy
    griffy           display comic horror playful
    fingerpaint      hand    comic horror
    lacquer          display comic horror
    # Sci-fi
    audiowide        display comic scifi
    orbitron         sans    comic scifi
    exo2             sans    comic scifi
    michroma         sans    comic scifi
    electrolize      sans    scifi
    zendots          display comic scifi
    novasquare       display comic scifi
    quantico         sans    comic scifi
    tektur           display comic scifi
    aldrich          sans    scifi
    iceland          display scifi
    wallpoet         display comic scifi
    monofett         display comic scifi retro
    sairastencilone  display comic pulp
    syncopate        sans    scifi
    chakrapetch      sans    comic scifi
    rajdhani         sans    scifi
    teko             sans    comic scifi pulp
    kanit            sans    comic pulp
    unbounded        display comic scifi
    syne             sans    scifi elegant
    spacegrotesk     sans    scifi
    # Fantasy and decorative
    cinzeldecorative display novel fantasy elegant
    almendradisplay  display novel fantasy
    macondo          display novel fantasy
    uncialantiqua    display novel fantasy
    medievalsharp    display novel fantasy
    metamorphous     display novel fantasy
    eaglelake        display novel fantasy
    aboreto          display novel elegant
    cinzel           serif   novel fantasy elegant
    berkshireswash   script  novel retro
    # Blackletter
    pirataone        blackletter novel horror fantasy
    grenzegotisch    blackletter novel horror fantasy
    unifrakturmaguntia blackletter novel horror fantasy
    unifrakturcook   blackletter novel horror fantasy
    fruktur          blackletter novel horror fantasy
    texturina        blackletter novel fantasy
    jacquard24       blackletter novel fantasy
    # Novel serif
    playfairdisplay  serif   novel elegant
    playfairdisplaysc serif  novel elegant
    cormorantgaramond serif  novel elegant
    cormorant        serif   novel elegant
    cormorantsc      serif   novel elegant
    cormorantunicase serif   novel fantasy
    ebgaramond       serif   novel elegant
    librebaskerville serif   novel elegant
    lora             serif   novel
    crimsontext      serif   novel
    merriweather     serif   novel
    oldstandardtt    serif   novel elegant
    spectral         serif   novel elegant
    prata            serif   novel elegant
    rozhaone         serif   novel elegant
    dmserifdisplay   serif   novel elegant
    yesevaone        serif   novel elegant
    ptserif          serif   novel
    sortsmillgoudy   serif   novel elegant
    ibmplexserif     serif   novel
    alegreya         serif   novel
    alegreyasc       serif   novel
    cardo            serif   novel
    vollkorn         serif   novel
    vollkornsc       serif   novel
    gildadisplay     serif   novel elegant
    bodonimoda       serif   novel elegant
    italiana         serif   novel elegant
    marcellus        serif   novel elegant fantasy
    marcellussc      serif   novel elegant fantasy
    forum            serif   novel elegant fantasy
    ledger           serif   novel
    imfelldwpica     serif   novel horror
    imfellenglish    serif   novel horror
    imfellenglishsc  serif   novel horror
    imfellfrenchcanon serif  novel horror
    imfelldoublepica serif   novel horror
    imfellgreatprimersc serif novel horror
    rosarivo         serif   novel
    fanwoodtext      serif   novel
    gelasio          serif   novel
    frankruhllibre   serif   novel
    lindenhill       serif   novel elegant
    goudybookletter1911 serif novel
    ovo              serif   novel elegant
    rufina           serif   novel
    mate             serif   novel
    anticdidone      serif   novel elegant
    abhayalibre      serif   novel
    castoro          serif   novel
    fraunces         serif   novel retro
    newsreader       serif   novel
    youngserif       serif   novel retro
    abrilfatface     serif   comic novel pulp
    gloock           serif   novel elegant
    bellefair        serif   novel elegant
    cinzel           serif   novel fantasy elegant
    trirong          serif   novel
    domine           serif   novel
    arapey           serif   novel elegant
    belgrano         slab    novel
    caudex           serif   novel fantasy
    ysabeau          sans    novel elegant
    # Slab
    robotoslab       slab    comic novel
    arvo             slab    novel pulp
    zillaslab        slab    novel
    bitter           slab    novel
    rokkitt          slab    novel pulp
    josefinslab      slab    novel retro
    aleo             slab    novel
    creteround       slab    novel
    patuaone         slab    comic pulp
    breeserif        slab    comic novel
    sanchez          slab    novel
    kreon            slab    novel
    copse            slab    novel
    podkova          slab    novel
    specialelite     slab    novel pulp
    courierprime     mono    novel pulp
    # Sans
    montserrat       sans    novel comic
    raleway          sans    novel elegant
    poppins          sans    novel
    worksans         sans    novel
    josefinsans      sans    novel retro elegant
    opensans         sans    novel
    lato             sans    novel
    roboto           sans    novel
    robotocondensed  sans    novel pulp
    barlowcondensed  sans    comic pulp
    sairacondensed   sans    comic scifi
    exo              sans    scifi
    ubuntu           sans    novel
    ubuntucondensed  sans    novel pulp
    ubuntumono       mono    novel scifi
    questrial        sans    novel
    lexend           sans    novel
    bigshouldersdisplay sans comic pulp
    leaguespartan    sans    comic novel
    leaguegothic     sans    comic pulp retro
    pathwaygothicone sans    novel pulp
    khand            sans    comic scifi
    saira            sans    scifi
    mavenpro         sans    novel
    tenorsans        sans    novel elegant
    belleza          sans    novel elegant
    federo           sans    novel retro
    poiretone        display novel retro elegant
    limelight        display novel retro elegant
    # Script and handwriting
    lobster          script  comic novel retro
    lobstertwo       script  novel retro
    pacifico         script  comic retro playful
    greatvibes       script  novel elegant
    dancingscript    script  novel
    satisfy          script  novel retro
    kaushanscript    script  comic novel
    yellowtail       script  comic retro
    sacramento       script  novel elegant
    allura           script  novel elegant
    alexbrush        script  novel elegant
    parisienne       script  novel elegant
    pinyonscript     script  novel elegant
    tangerine        script  novel elegant
    cookie           script  novel retro
    courgette        script  novel
    damion           script  comic retro
    marckscript      script  novel
    mrdafoe          script  comic novel
    rochester        script  novel retro
    monsieurladoulaise script novel elegant
    kalam            hand    novel playful
    caveat           hand    novel playful
    shadowsintolight hand    novel playful
    indieflower      hand    novel playful
    gochihand        hand    comic playful
    rocksalt         hand    comic playful
    homemadeapple    hand    novel
    calligraffitti   hand    comic playful
    justanotherhand  hand    comic
    walterturncoat   hand    comic playful
    unkempt          hand    comic playful
    kranky           display comic playful
    mountainsofchristmas display comic fantasy playful
    """
)


def unique_families() -> list[FamilyEntry]:
    seen: dict[str, FamilyEntry] = {}
    for entry in FAMILIES:
        seen.setdefault(entry.dirname, entry)
    return list(seen.values())
