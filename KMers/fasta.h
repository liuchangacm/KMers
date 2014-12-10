#pragma once

class FastaReader {
public:
	bool next();
	char* getKey();
	char* getValue();
	bool open(const char *);
};
