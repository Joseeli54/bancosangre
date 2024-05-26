import {Entity} from '../classes/entity'

export class User extends Entity{
    constructor(
        public name? : string,
        public apellido? : string,
        public username? : string,
        public email? : string,
        public role? : string,
        public avatar? : string,
        public avatar_public_id? : string,
        public type? : string,
        public direccion? : string,
        public telefono? : string,
    ){
        super();
    }
}