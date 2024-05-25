import {Entity} from '../classes/entity'

export class Diagnostico extends Entity{
    constructor(
        public name? : string,
        public description? : string,
        public id_sangre? : string,
        public id_user? : string,
        public createBy? : string
    ){
        super();
    }
}