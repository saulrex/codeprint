import React from 'react';
import lodash from 'lodash';
import { helper } from './helpers';
import { api } from './api';

export function processData(inputData) {
  const glyphPrefix = inputData.map((item) => {
    return item.value * 2;
  });

  const filteredData = glyphPrefix.filter((x) => x > 10);

  const destinationDocuments = filteredData.length;
  return { data: filteredData, count: destinationDocuments };
}
