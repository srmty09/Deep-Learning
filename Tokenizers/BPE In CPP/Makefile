########################################################################
####################### Makefile Template ##############################
########################################################################

# Compiler settings - Can be customized.
CC = g++
CXXFLAGS = -std=c++11 -Wall
LDFLAGS = 

# Makefile settings - Can be customized.
APPNAME = main
APINAME = bpe_api
SIMPLEAPI = simple_api
EXT = .cpp
SRCDIR = src
OBJDIR = obj

############## Do not change anything from here downwards! #############
# Source files (excluding main programs)
COMMON_SRC = $(SRCDIR)/bpe.cpp $(SRCDIR)/filereading.cpp
MAIN_SRC = $(SRCDIR)/main.cpp
API_SRC = $(SRCDIR)/bpe_api.cpp
SIMPLE_API_SRC = $(SRCDIR)/simple_api.cpp

# Object files
COMMON_OBJ = $(COMMON_SRC:$(SRCDIR)/%$(EXT)=$(OBJDIR)/%.o)
MAIN_OBJ = $(MAIN_SRC:$(SRCDIR)/%$(EXT)=$(OBJDIR)/%.o)
API_OBJ = $(API_SRC:$(SRCDIR)/%$(EXT)=$(OBJDIR)/%.o)
SIMPLE_API_OBJ = $(SIMPLE_API_SRC:$(SRCDIR)/%$(EXT)=$(OBJDIR)/%.o)

ALL_OBJ = $(COMMON_OBJ) $(MAIN_OBJ) $(API_OBJ) $(SIMPLE_API_OBJ)
DEP = $(ALL_OBJ:$(OBJDIR)/%.o=%.d)

# UNIX-based OS variables & settings
RM = rm
DELOBJ = $(ALL_OBJ)
# Windows OS variables & settings
DEL = del
EXE = .exe
WDELOBJ = $(ALL_OBJ:$(OBJDIR)/%.o=$(OBJDIR)\\%.o)

########################################################################
####################### Targets beginning here #########################
########################################################################

all: $(APPNAME) $(SIMPLEAPI)

# Builds the main app
$(APPNAME): $(COMMON_OBJ) $(MAIN_OBJ)
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Builds the API app
$(APINAME): $(COMMON_OBJ) $(API_OBJ)
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Builds the simple API app
$(SIMPLEAPI): $(COMMON_OBJ) $(SIMPLE_API_OBJ)
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Creates the dependecy rules
%.d: $(SRCDIR)/%$(EXT)
	@$(CPP) $(CFLAGS) $< -MM -MT $(@:%.d=$(OBJDIR)/%.o) >$@

# Includes all .h files
-include $(DEP)

# Building rule for .o files and its .c/.cpp in combination with all .h
$(OBJDIR)/%.o: $(SRCDIR)/%$(EXT)
	$(CC) $(CXXFLAGS) -o $@ -c $<

################### Cleaning rules for Unix-based OS ###################
# Cleans complete project
.PHONY: clean
clean:
	$(RM) $(DELOBJ) $(DEP) $(APPNAME) $(APINAME)

# Cleans only all files with the extension .d
.PHONY: cleandep
cleandep:
	$(RM) $(DEP)

#################### Cleaning rules for Windows OS #####################
# Cleans complete project
.PHONY: cleanw
cleanw:
	$(DEL) $(WDELOBJ) $(DEP) $(APPNAME)$(EXE) $(APINAME)$(EXE)

# Cleans only all files with the extension .d
.PHONY: cleandepw
cleandepw:
	$(DEL) $(DEP)