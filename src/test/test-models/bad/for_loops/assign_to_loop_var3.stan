transformed data {
  // nested foreach (array)
  while (1) {
    int vs[2, 3];
    for (v in vs) {
      v[1] = 0;
      break;
    }        
  }

}
